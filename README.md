# alive

### Turn any creator into an AI clone you can talk to.

Feed it their YouTube videos. Ask it anything. Watch them answer, in their own words,
voice and face.

```mermaid
flowchart LR
    Q([Your question]) --> A["<b>Their words</b><br/>found in what<br/>they actually said"]
    A --> V["<b>Their voice</b><br/>learned from hours<br/>of their speech"]
    V --> F["<b>Their face</b><br/>moving the way<br/>theirs moves"]
    F --> W([They answer])
```

- **From their own words.** Answers come from their videos, and every claim points back to the moment they said it.
- **Sounds like them.** Their voice, their pace, their pitch.
- **Looks like them.** Their face, drawn fresh onto real footage of their body.
- **Any creator.** Give it their videos, get their clone.

> Every clone is AI-generated and labelled so. Ask a creator before you clone them.

## Requirements

| | to run a clone | to build one |
|---|---|---|
| **GPU** | NVIDIA, 12 GB | NVIDIA, 16 GB (built on an RTX 5080) |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 3 GB per clone | 150 GB while building |
| **Time** | starts in 20 s | about a day per creator |

**Software:** Linux, an NVIDIA driver for CUDA 12.8, Miniconda, uv, ffmpeg, yt-dlp.

**Bring your own** (these can't be shared):
- an [Anthropic API key](https://console.anthropic.com), for writing the answers
- the [FLAME](https://flame.is.tue.mpg.de) head model, free for research
- NVIDIA's Audio2Face-3D models
- a MetaHuman face from Epic's MetaHuman Creator

[Setup](https://pavanteja295.github.io/alive/setup.html) walks through each one.

## Try it

```bash
./alive build huberman     # from YouTube to trained models, about a day
./alive up huberman        # then open http://127.0.0.1:8800
```

**[Read the guide →](https://pavanteja295.github.io/alive/)** for setup, how each part
works, and how to add your own creator.

Built on [STAvatar](https://github.com/JiankuoZhao/STAvatar), [F5-TTS](https://github.com/SWivid/F5-TTS),
[VHAP](https://github.com/ShenhanQian/VHAP), NVIDIA Audio2Face-3D, Epic MetaHuman and
Anthropic Claude ([credits](https://pavanteja295.github.io/alive/credits.html)).
