# # python批量更换后缀名
# import os
#
# # 列出当前目录下所有的文件
# files = os.listdir('C:/Users/Administrator/PycharmProjects/xiebaolai/AAA/data/CK+224/contempt/')
# for filename in files:
# 	portion = os.path.splitext(filename)
# 	# 如果后缀是.dat
# 	if portion[1] == ".jpg":
# 		# 重新组合文件名和后缀名
# 		newname = portion[0]
# 		os.rename(filename,newname)


# python批量更换后缀名
import os
import sys
os.chdir(r'C:/Users/Administrator/PycharmProjects/xiebaolai/AAA/data/CK+224/contempt/')

# 列出当前目录下所有的文件
files = os.listdir('shu_image_cm/')
print('files',files)

for fileName in files:
	portion = os.path.splitext(fileName)
	# 如果后缀是.dat
	if portion[1] == ".jpg":
		#把原文件后缀名改为 txt
		newName = portion[0]
		os.rename(fileName, newName)

