import discord
from discord.ext import commands
from SECRETS import bot_token
import pandas as pd
import torch as pt
from tokenizers import Tokenizer
import h5py as hp
from utils import Embedder, pca
import pandas as pd
from model import Model

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
device = "cpu"
RWKV_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
TOKENIZER_FILE = "../model/20B_tokenizer.json"
N_LAYER = 24
N_EMBD = 1024
model_file = "model.ckpt"

print(f"Loading {TOKENIZER_FILE} and {RWKV_FILE}...")
embed = Embedder(TOKENIZER_FILE, RWKV_FILE, N_LAYER)

print(f"Loading {model_file}...")
model = Model(features=[1024, 1024, 1024, 512, 128, 32, 5]).to(device)
#model = Model(features=[1024, 512, 32, 5]).to(device)
weights = pt.load(model_file)
model.load_state_dict(weights)
model.eval()

labels = ["Negative spike", "Negative followed by positive spike", "Positive followed by negative spike", "Positive spike", "No meaningful spikes"]
emojis = [":chart_with_downwards_trend:", ":chart_with_downwards_trend: :chart_with_upwards_trend:", ":chart_with_upwards_trend: :chart_with_downwards_trend:", ":chart_with_upwards_trend:", "🗠"]


intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.members = True
intents.guilds = True
client = commands.Bot(command_prefix="/", intents=intents)


@client.event
async def on_ready():
    print(f"{client.user} is active!")
    channel = client.get_channel(1323816118959341618)
    await channel.send("> \"We at Hooli believe that data is the lifeblood of the modern economy, coursing through the veins of global commerce. We at Hooli Insights: Mapping that flow, predicting its currents, and harnessing its power for your financial gain. Welcome to Hooli Insights.\"\n> \\- Gavin Christ")


@client.command()
async def read(ctx, *, text):
    print(f"{ctx.author.name} in {ctx.channel.id} requested:\n\t{text}")

    emb = embed([text])
    pred = model(emb)[0]

    ans = pred.argmax().item()
    response = f"# {emojis[ans]} {labels[ans]} {pred[ans] * 100:.4f}%"
    for i, p, l in zip(range(len(labels)), pred, labels):
        if i == ans:
            response += f"\n- **{l}: {p * 100:.4f}%**"
        else:
            response += f"\n- {l}: {p * 100:.4f}%"
    await ctx.reply(response)


client.run(bot_token)
