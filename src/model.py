import torch as pt


lin = lambda w, x: pt.tensordot(x, w, dims=[[-1], [-1]])


class LayerNorm(pt.nn.Module):

    def __init__(self, w, b):

        super().__init__()
        self.w = w
        self.b = b

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / std
        x = x * self.w + self.b
        
        return x


class TimeMix(pt.nn.Module):

    def __init__(self, decay, bonus, mix_k, mix_v, mix_r, w_k, w_v, w_r, w_out, last_x=0, last_num=0, last_den=0):

        super().__init__()
        self.decay = decay
        self.bonus = bonus
        self.mix_k = mix_k
        self.mix_v = mix_v
        self.mix_r = mix_r
        self.w_k = w_k
        self.w_v = w_v
        self.w_r = w_r
        self.w_out = w_out
        self.last_x = last_x
        self.last_num = last_num
        self.last_den = last_den
        self.sigmoid = pt.nn.Sigmoid()

    def reset(self):

        self.last_x = 0
        self.last_num = 0
        self.last_den = 0

    def forward(self, x):

        last_x = pt.zeros(x.shape, dtype=x.dtype, device=x.device)
        last_x[..., 1:, :] = x[..., :-1, :]
        last_x[..., 0, :] = self.last_x
        self.last_x = x[..., -1, :]

        k = lin(self.w_k, x * self.mix_k + last_x * (1 - self.mix_k))
        v = lin(self.w_v, x * self.mix_v + last_x * (1 - self.mix_v))
        r = lin(self.w_r, x * self.mix_r + last_x * (1 - self.mix_r))

        self.wkv = pt.zeros(k.shape)
        for i in range(k.shape[-2]):
            exp_bonus_k = pt.exp(self.bonus + k[..., i, :])
            self.wkv[..., i, :] = (self.last_num + exp_bonus_k * v[..., i, :]) / (self.last_den + exp_bonus_k)
            decay = pt.exp(-pt.exp(self.decay))
            exp_k = pt.exp(k[..., i, :])
            self.last_num = decay * self.last_num + exp_k * v[..., i, :]
            self.last_den = decay * self.last_den + exp_k

        rwkv = self.sigmoid(r) * self.wkv

        return lin(self.w_out, rwkv)


class ChannelMix(pt.nn.Module):

    def __init__(self, mix_k, mix_r, w_k, w_r, w_v, last_x=0):

        super().__init__()
        self.mix_k = mix_k
        self.mix_r = mix_r
        self.w_k = w_k
        self.w_k = w_k
        self.w_r = w_r
        self.w_v = w_v
        self.last_x = last_x
        self.relu = pt.nn.ReLU()
        self.sigmoid = pt.nn.Sigmoid()

    def reset(self):

        self.last_x = 0

    def forward(self, x):

        last_x = pt.zeros(x.shape, dtype=x.dtype, device=x.device)
        last_x[..., 1:, :] = x[..., :-1, :]
        last_x[..., 0, :] = self.last_x
        self.last_x = x[..., -1, :]

        k = lin(self.w_k, x * self.mix_k + last_x * (1 - self.mix_k))
        r = lin(self.w_r, x * self.mix_r + last_x * (1 - self.mix_r))
        vk = lin(self.w_v, self.relu(k) ** 2)

        return self.sigmoid(r) * vk


class RWKV(pt.nn.Module):

    def __init__(self, state_dict, layer_amnt):

        super().__init__()
        params = lambda prefix : [state_dict[key] for key in state_dict.keys() if key.startswith(prefix)]
        self.emb = params("emb")[0]
        self.ln_in = LayerNorm(*params("blocks.0.ln0"))
        self.ln1 = [LayerNorm(*params(f"blocks.{i}.ln1")) for i in range(layer_amnt)]
        self.att = [TimeMix(*params(f"blocks.{i}.att")) for i in range(layer_amnt)]
        self.ln2 = [LayerNorm(*params(f"blocks.{i}.ln2")) for i in range(layer_amnt)]
        self.ffn = [ChannelMix(*params(f"blocks.{i}.ffn")) for i in range(layer_amnt)]

    def reset(self):

        for a in self.att:
            a.reset()

        for f in self.ffn:
            f.reset()

    def forward(self, x, lengths=-1):

        x = self.emb[x]
        x = self.ln_in(x)

        for ln1, att, ln2, ffn in zip(self.ln1, self.att, self.ln2, self.ffn):
            x += att(ln1(x))
            x += ffn(ln2(x))

        state = att.wkv[pt.arange(len(x)), lengths, :]

        return state


class Model(pt.nn.Module):

    def __init__(self, in_features=1024, out_features=2):

        super().__init__()
        self.linear = pt.nn.Linear(in_features=in_features, out_features=out_features)
        self.softmax = pt.nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.linear(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("../model/20B_tokenizer.json")
    state_dict = pt.load("../model/RWKV-4-Pile-430M-20220808-8066.pth", map_location="cpu")
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].squeeze()
        state_dict[k] = state_dict[k].float()

    m = RWKV(state_dict, 24)
    x = pt.tensor(tokenizer.encode("hello how are you").ids)[None, :]
    y = m(x)

