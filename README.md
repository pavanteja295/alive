# alive

### Ask a creator anything. Watch them answer, in their own words, voice and face.

Point it at a creator's YouTube videos. It learns how they talk, sound and move, and
builds a live version of them you can question.

```mermaid
flowchart LR
    Q([Your question]) --> A[Their words] --> V[Their voice] --> F[Their face] --> W([They answer])
```

> Everything it shows is AI-generated and labelled so. Ask a creator before you build one of them.

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
