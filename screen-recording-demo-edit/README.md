# 录屏演示视频剪辑

把一段产品或 App 的操作录屏剪成演示视频：用户点到哪个功能，就平滑放大哪一块，配一句讲解和标签。讲解时压低原声，其余时候保留原声。

## 依赖

- ffmpeg
- Python，以及 numpy、opencv-python、Pillow
- edge-tts
- fonts-noto-cjk

## 怎么用

把本目录的 `SKILL.md` 交给 agent。用户提供录屏后，agent 按里面的步骤核对时间线、写剪辑决策、生成配音并渲染成片。

## 脚本

`scripts/build_demo.py` 是针对某一段具体录屏写好的参考实现。里面的片段、放大窗口、点击和配音时间都只对那一段有效，换视频要重新测量。它还依赖工作目录里由 edge-tts 生成的讲解音频和 `lines.json`。
