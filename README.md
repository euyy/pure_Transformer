<!--
 * @Description: 
 * @version: 
 * @Author: Yue Yang
 * @Date: 2022-03-23 17:01:38
 * @LastEditors: Yue Yang
 * @LastEditTime: 2022-04-21 16:04:32
-->

# 构想
使用Transformer实现有条件的从文本生成图像任务。
# 模型
首先参照AttnGAN修改，由生成器 $G$ 和判别器 $D$ 构成。

生成器 $G$ 由文本编码器，图像解码器和图像编码器构成。三者都是通过Transformer实现的。图像编码器可能会需要一些卷积操作进行图像特征的获取。

判别器 $D$ 由图像编码器实现，首位的输出True/False作为判别图像真假的结果，类似于Bert。

- 生成器 $G$ :
  - 文本编码器: text encoder(TE)
  - 图像解码器: image decoder(ID_G)
  - 图像编码器: image encoder(IE)

- 判别器 $D$ :
  - 图像编码器: image decoder(ID_D)

# 训练运行的命令
python main.py --cfg cfg/coco_train.yml --gpu 0
# 2022/04/06 
看看AttnGAN的D_GET_LOGITS是怎么计算的，计算的结果是一个数还是一个向量
看看二分类的论文，最后是如何输出的，一个一位向量如何映射成一个logits值

# 2022/04/12
基本写完了模型的训练阶段，开始修改bug

# 2022/04/21
完成了出版本的模型，模型包括文本编码器，图像解码器，图像生成器，图像编码器和判别器。
**文本编码器**使用Transformer的encoder结构
**图像解码器**使用Transformer的decoder结构
**图像生成器**采用了AttnGAN的生成器结构，但是只有一阶段，并且没有对图像进行上采样
**图像编码器**使用了Transformer的encoder结构，输入图像和文本，并将输出用于计算words_loss
**判别器**使用了Transformer的encoder结构，