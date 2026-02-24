// For more customizable options, please refer to official reference: https://typst.app/docs/reference/

#show heading: set text(font: "FZShusong-Z01")

#set text(font: "FZShusong-Z01")

#show link: underline

// Uncomment the following lines to adjust the size of text
// The recommend resume text size is from `10pt` to `12pt`
// #set text(
//   size: 12pt,
// )

// Feel free to change the margin below to best fit your own CV
#set page(
  margin: (x: 0.9cm, y: 1.3cm),
)


#set par(justify: true)

#let chiline() = {
  v(-3pt)
  line(length: 100%)
  v(-5pt)
}

#let continuescvpage() = {
  place(
    bottom + center,
    dx: 0pt, // Horizontal offset (positive is rightward)
    dy: -10pt, // Vertical offset (positive moves upwards)
    float: true,
    scope: "parent",
    [
      #text(fill: gray)[... continues on the next page ...]
    ],
  )
}

#let lastupdated(date) = {
  h(1fr)
  text("Last Updated in " + date, fill: color.gray)
}

// Uncomment the following lines to add the optional prompt at the bottom of the first CV page
// #continuescvpage()

= 左元

zuoyuantc\@gmail.com |
#link("https://github.com/confucianzuoyuan")[github.com/confucianzuoyuan] |
18518538812

== 教育经历
#chiline()

中国科学院电子学研究所 #h(1fr) 2008/09 -- 2011/07 \
电子科学与技术 #h(1fr) 工学硕士 \

中国农业大学 #h(1fr) 2004/09 -- 2008/07 \
测控技术与仪器 #h(1fr) 工学学士 \

== 工作经历
#chiline()

尚硅谷IT培训机构 #h(1fr) 2017/12 -- 现在 \
职位：研发 & 讲师
- 研发并讲解强化学习课程，被多名学员评为最好的强化学习课程，大幅领先网络上的所有其它课程，特点如下：
  - 以策略梯度法贯穿始终：原始策略梯度法 #sym.arrow REINFORCE #sym.arrow Actor-Critic #sym.arrow 置信域策略优化（TRPO） #sym.arrow 近端策略优化（PPO） #sym.arrow 组相对策略优化（GRPO）
  - 注重算法的数学细节，例如近端策略优化（PPO）算法的数学推导为全网最为详细的推导和讲解
  - 注重代码实现细节，所有算法的实现全部#underline([手搓])，不做调包侠和调参侠（不调用transofmers和trl等库）。
  - 从零复刻了InstructGPT的实现
  - 从零复刻了DeepSeek-R1的训练流程
- 研发并讲解多模态课程，被多名学员评为最好的强化学习课程，大幅领先网络上的所有其它课程，特点如下：
  - 注重算法的数学细节，例如严格推导了DDPM、得分函数等扩散模型相关算法。
  - 注重代码是实现细节，所有算法全部#underline([手搓])。
  - 从零实现Vision Transformer模型。
  - 从零实现并预训练CLIP模型。
  - 从零复刻了ClipCap论文，实现"图生文"功能。
  - 从零实现DDPM算法以及条件扩散模型和无分类器指引的扩散模型。
  - 从零复刻了Dall-E2（训练CLIP模型，训练Prior Model，训练扩散解码器模型），实现"文生图"功能。

- #link("https://github.com/confucianzuoyuan/zcc")[zcc]：一个完整支持C11语法的C语言编译器
  - 从头使用Golang实现，未使用任何依赖
  - 可以使用zcc编译CPython，SQLite，Lua等著名开源项目
  - 实现了预处理器，词法分析，语法分析，语义分析，代码生成（目前只支持x86-64指令集）等功能。
  - 由于只做了简单的优化（指令选择，强度削减等），所以编译CPython等开源项目的速度比gcc编译器快4 #sym.tilde 5倍

- #link("https://github.com/confucianzuoyuan/PyZorch")[PyZorch]：一个类似PyTorch的自动微分框架
  - 张量计算部分使用C++ CUDA实现，并编译为动态链接库"libtensor.so"供python调用。
  - 使用python实现计算图的构建和反向传播算法。

- #link("https://github.com/confucianzuoyuan/tiger-rust")[tiger-rust]：Tiger语言编译器
  - Tiger语言为著名编译器教材《现代编译原理》（"虎书"）中设计的编译型编程语言
  - 支持高阶函数功能
  - 使用Rust实现Tiger语言编译器。
  - 多级IR设计（Tree IR #sym.arrow Linear IR #sym.arrow ASM）
  - 实现了图着色寄存器分配算法
