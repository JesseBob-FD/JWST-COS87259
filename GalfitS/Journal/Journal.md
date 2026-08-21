# 6.19
GalfitS/src/data/filters: What's the use of this data? Moreover, there are no filters for F560W, F1130W and F2550W. Does this mean that GalfitS cannot directly processing datas of those missing bands unless I creates corresponding filters?

The calculations of Convention factors seem to be more complicated than what I have imagined.

# 6.20
**Possible bug in the source codes**
Na27 should be a list of lists (as example in GalfitS/examples/data_example/obj29489_s1.lyric shows). However, examples in GalfitS/examples confused Na27 as a sigle list.

# 7.15
编译文件中的I*15是决定是否采用SED综合拟合的参数。When set to 0, shifts GalfitS to a pure photometric tool, akin to GALFIT. By default, it is set to 1, which employs SED information for fitting multi-band images.

# 7.16 Discussion with RC Li
1. five-value array: [initial value, minimum value, maximum value, typical variation step, fixed or not]. Choose 0 for the fifth means that the value is fixed at the intial value and it doesn't change in the fitting process?
2. 有filter文件的几个波段我尝试跑了几种方案做拟合，比如单sersic和两个sersic成分拟合，但是出来的结果（我认为）差别不大
3. The image fitting result of SED mode is quete different with no-SED mode (I*15 = 0).
4. GNz7q那篇文章的图四：offset在哪里？

## 建议：
1. cutoff范围改小一点
2. 把miri数据和nircam数据放在一起跑一下
3. 补充的filter文件已经在git上了   

# 7.22
目前来看，加入AGN的拟合从光谱上来看更合理，长波处明显v上去。

# 7.31
反思：针对使用Free AGN模型来拟合，从powerlaw模型换到free模型除了让SED的拟合看上去更加贴合之外并没有实质性的好处，实际上image的拟合结果中仍显著有中心的深蓝residual，说明拟合处的model强度过高了。参见host_agn_powerlaw和host_freeagn的比较。而后续固定中心的拟合中均采用free AGN模型，暂时也不用换回powerlaw，因之二者并没有在前序工作中展现出差距。

# 8.18
GPT指出数据预处理管线preprocess_nircam.py可能存在问题。其中一些问题，例如world_to_pixel_values不应额外减去一像素，PSF应用实测等，miri数据预处理管线也同样存在。需要一起验证。
world_to_pixel_values在代码中是将源中心的天球坐标转换为像素，使用world_to_pixel_values函数；使用pixel_to_world_values函数进行反变换并验证结果是否回到原天球坐标应该是检验的可行方法。