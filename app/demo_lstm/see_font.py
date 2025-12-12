import matplotlib.font_manager as fm

# 遍历所有字体，筛选出包含中文相关关键词的字体
font_list = [f.name for f in fm.fontManager.ttflist]
# Mac系统常见的中文字体关键词：Heiti、Arial Unicode MS、PingFang（苹方）等
chinese_fonts = [font for font in font_list if any(keyword in font for keyword in ["Heiti", "Unicode", "PingFang", "Song", "Kai"])]
# 去重并打印
print("你的系统可用中文字体：", list(set(chinese_fonts)))