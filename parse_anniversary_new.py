from PIL import Image
import pytesseract
from pytesseract import Output
import numpy as np
import argparse
import cv2, os, re, io, math, requests
import discord

AUCTION_RED_BOX = np.array([225, 79, 61])
AUCTION_RED_BOX_THRESH = np.array([40, 40, 20])

# revenue array of selling prizes for coins
rev = np.array(
	[
		[  360,120,60,36,9,9,6,360],  #index 0, ticket x1 160 prizes
		[  270,90,45,27,9,9,6,270],  #index 1, ticket x1 120 prizes
		[  360,180,120,90,18,18,12,360],  #index 2, ticket x2 80 prizes
		[  540,270,180,135,27,27,18,540],  #index 3, ticket x3 80 prizes
		[  810,270,135,81,27,27,18,810],  #index 4, ticket x3 120 prizes
		[  540,360,180,120,27,27,18,540]  #index 5, ticket x3 160 prizes
	]
)

points_values = { 
	"2_120": {"A": 0, "B": 180, "C": 90, "D": 54, "E": 18, "F": 18, "G": 12}, 
	"2_80":  {"A": 0, "B": 180, "C": 120, "D": 90, "E": 18, "F": 18, "G": 12}, 
	"3_80":  {"A": 0, "B": 270, "C": 180, "D": 135, "E": 27, "F": 27, "G": 18}, 
	"2_240": {"A": 0, "B": 270, "C": 120, "D": 72, "E": 36, "F": 12, "G": 12}, 
	"3_160": {"A": 0, "B": 360, "C": 180, "D": 120, "E": 27, "F": 27, "G": 18}, }

rev = np.array(
	[
		[  0,0,0,0,0,0,0,0],  #index 0, old chest
		[  540,180,90,54,18,18,12,540],  #index 1 ticket x2 120 prizes ** @ checked smaller tech/s chest
		[  540,270,180,135,27,27,18,540],  #index 2, ticket x3 80 prizes ** @ checked taxola
		[  360,180,120,90,18,18,12,360],  #index 3, ticket x2 80 prizes ** @ checked outfits
		[  540,270,120,72,36,12,12,540],  #index 4, ticket x2 240 prizes ** @ checked rc/ac/reschip chest
		[  540,360,180,120,27,27,18,540]  #index 5, ticket x3 160 prizes ** @ checked core selector
	]
)

def filter_image(image,color,threshold):
	# filter image by color +/- thresh, returns mask
	if image.shape[2]==4:
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

	lowthresh = color - threshold
	highthresh = color + threshold
	img_inrange = cv2.inRange(image,lowthresh,highthresh)
	img_inrange = cv2.bitwise_not(img_inrange)
	return img_inrange

def largest_bounding_box(image):
	# input image is white bg black shapes
	# f(x) inverts to white shapes on black bg, find largest bounding box
	inverted_not = cv2.bitwise_not(image)
	contours, _ = cv2.findContours(inverted_not,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	c = max(contours, key = cv2.contourArea)
	x,y,w,h = cv2.boundingRect(c)

	top,bot,left,right = y, y+h, x, x+w
	print(f"(top,left),(bot,right): ({top},{left}),({bot},{right})")
	return top,bot,left,right

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def crop_image(img,left,right,top,bottom):
	cropped = img[round(img.shape[0]*top):round(img.shape[0]*bottom),round(img.shape[1]*left):round(img.shape[1]*right)]
	ratio = cropped.shape[0]/cropped.shape[1]
	y=int(200*ratio)
	cropped = cv2.resize(cropped, (200,y), interpolation = cv2.INTER_CUBIC)
	if np.mean(cropped)>250:
		print("blank detected, printing 0 image")
		cv2.putText(cropped, '0/0', (60,round(y/1.5)), cv2.FONT_HERSHEY_SIMPLEX,  1.3, (0,0,0), 4, cv2.LINE_AA)

	cv2.fastNlMeansDenoising(cropped, cropped, 30, 7, 21)

	row, col = cropped.shape[:2]
	bottom = cropped[row-2:row, 0:col]
	mean = cv2.mean(bottom)[0]

	border_size = 20
	cropped = cv2.copyMakeBorder(
	    cropped,
	    top=border_size,
	    bottom=border_size,
	    left=border_size,
	    right=border_size,
	    borderType=cv2.BORDER_CONSTANT,
	    value=[mean, mean, mean]
	)
	return cropped

def parse_image(image):
	debug_files=[]
	error_text = None
	response = requests.get(image)
	img = Image.open(io.BytesIO(response.content))
	# load the example image
	print("loading images from attachment")
	img = np.array(img)

	imgwidth=img.shape[1]
	imgheight=img.shape[0]

	red_box_filtered = filter_image(img, AUCTION_RED_BOX,AUCTION_RED_BOX_THRESH)
	inverted = cv2.bitwise_not(red_box_filtered)

	contours, _ = cv2.findContours(inverted,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	c = max(contours, key = cv2.contourArea)
	x,y,w,h = cv2.boundingRect(c)
	top, bot, left, right = y, y+h, x, x+w

	scanimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	scanimg = cv2.threshold(scanimg, 6, 255, cv2.THRESH_BINARY)[1]

	initial_pix = scanimg[round(imgheight*.4),1]
	end_x=0
	for x in range(1,imgwidth):
		this_pix = scanimg[round(imgheight*.4),x]
		if np.array_equal(initial_pix,this_pix):
			end_x=x
		else:
			break

	### find top
	end_x=end_x+10
	initial_pix=scanimg[round(imgheight*.4),end_x]
	top_y=round(imgheight*.4)
	for y in range(round(imgheight*.4),0,-1):
		this_pix = scanimg[y,end_x]
		if np.array_equal(initial_pix,this_pix):
			top_y=y
			#img[y,end_x+5]=(0,0,255)
		else:
			break

	### find bottom
	initial_pix=scanimg[round(imgheight*.4),end_x]
	bot_y=round(imgheight*.4)
	for y in range(round(imgheight*.4),imgheight):
		this_pix = scanimg[y,end_x]
		if np.array_equal(initial_pix,this_pix):
			bot_y=y
			#img[y,end_x+5]=(0,0,255)

		else:
			break

	# crop1 = img[top:bot,left:right]
	crop1 = img[top_y:bot_y,end_x:(imgwidth-end_x)]


	crop1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2GRAY)
	crop1 = cv2.threshold(crop1, 215, 255, cv2.THRESH_BINARY_INV)[1]
	#crop1 = cv2.resize(crop1, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

	x=crop1.shape[1]
	y=crop1.shape[0]

	#crop2 passed to image process, crop1 passed to draw crop rectangles
	crop2 = np.array(crop1)
	crop1 = cv2.cvtColor(crop1,cv2.COLOR_GRAY2RGB)

	crop_width=0.14		# 14% of crop1 width
	crop_height=0.05	#  5% of crop1 height

	left=0.15
	top=0.4
	right=left+crop_width
	bottom=top+crop_height
	crop1 = cv2.rectangle(crop1, (int(x*left), int(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	A=crop_image(crop2,left,right,top,bottom)

	left=0.47
	top=0.31
	right=left+crop_width
	bottom=top+crop_height
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	B=crop_image(crop2,left,right,top,bottom)

	left= 0.75
	top= 0.31
	right=left+crop_width
	bottom=top+crop_height
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	C=crop_image(crop2,left,right,top,bottom)

	left= 0.47
	top= 0.53
	right=left+crop_width
	bottom=top+crop_height
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	D=crop_image(crop2,left,right,top,bottom)

	left= 0.75
	top= 0.53
	right=left+crop_width
	bottom=top+crop_height
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	E=crop_image(crop2,left,right,top,bottom)

	left= 0.47
	top= 0.75
	right=left+crop_width
	bottom=top+crop_height
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	F=crop_image(crop2,left,right,top,bottom)

	left= 0.75
	top= 0.75
	right=left+crop_width
	bottom=top+crop_height
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	G=crop_image(crop2,left,right,top,bottom)

	left=0.15
	top=0.74
	right=left+crop_width
	bottom=top+crop_height
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	L=crop_image(crop2,left,right,top,bottom)

	left=0.92
	top=0.02
	right=left+.06
	bottom=top+.06
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	topPrice=crop_image(crop2,left,right,top,bottom)

	left=0.49
	top=0.9
	right=left+.09
	bottom=top+.06
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	botPrice=crop_image(crop2,left,right,top,bottom)

	left=0.47
	top=0.08
	right=left+.32
	bottom=top+.05
	crop1 = cv2.rectangle(crop1, (round(x*left), round(y*top)), (round(x*right), round(y*bottom)),(0, 255, 0), 2) #Plotting bounding box
	rewardsLeft=crop_image(crop1,left,right,top,bottom)

	# stack all prize crops
	append_img = np.append(A,L,0)
	append_img = np.append(append_img,B,0)
	append_img = np.append(append_img,C,0)
	append_img = np.append(append_img,D,0)
	append_img = np.append(append_img,E,0)
	append_img = np.append(append_img,F,0)
	append_img = np.append(append_img,G,0)

	topPrice = cv2.resize(topPrice, None,fx=0.5,fy=0.5, interpolation = cv2.INTER_CUBIC)
	botPrice = cv2.resize(botPrice, None,fx=0.5,fy=0.5, interpolation = cv2.INTER_CUBIC)        

	tessdata_dir_config = (
	    f'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata" -l eng --psm 6'
	)
	text = pytesseract.image_to_string(image=rewardsLeft,config=tessdata_dir_config)
	print(f"rewardsLeft scanned text :\n{text}")
	text=re.findall(r"\d{1,3}",text)
	rewardsLeft=int(text[0])

	tessdata_dir_config = (
	    f'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata" -c tessedit_char_whitelist="/xX0123456789" -l eng --psm 6 --user-patterns aniv_patterns'
	)
	text = pytesseract.image_to_string(image=append_img,config=tessdata_dir_config)
	#print(f"append scanned text :\n{text}")
	text=re.findall(r"\d{1,2}/\d{1,2}",text)
	#print(f"re.findall :\n {text}")
	if len(text)!=8:
		tessdata_dir_config = (
			f'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata" -c tessedit_char_whitelist="/xX0123456789" -l eng --psm 3 --user-patterns aniv_patterns'
		)
	text = pytesseract.image_to_string(image=append_img,config=tessdata_dir_config)
	text = re.findall(r"\d{1,2}/\d{1,2}",text)

	if len(text)!=8:
		return None, debug_files, f"Got less than 8 prizes"

	prizeA=int(text[0].split('/')[0])
	prizeL=int(text[1].split('/')[0])
	prizeB=int(text[2].split('/')[0])
	prizeC=int(text[3].split('/')[0])
	prizeD=int(text[4].split('/')[0])
	prizeE=int(text[5].split('/')[0])
	prizeF=int(text[6].split('/')[0])
	prizeG=int(text[7].split('/')[0])


	text = pytesseract.image_to_string(image=botPrice,config=tessdata_dir_config)
	print(f"botPrice text : {text}")
	text=re.findall(r"[xX][1-3]",text)
	if len(text)==1:
		xticket =int(text[0][1:])
	else:
		text = pytesseract.image_to_string(image=topPrice,config=tessdata_dir_config)
		#print(f"topPrice text : {text}")
		text=re.findall(r"[xX][1-3]",text)
		#print(f"re.findall :\n {text}")
		if len(text)==1:
			xticket =int(text[0][1:])
		else:
			return None, debug_files, f"Couldn't get xTicket"

	checked_text = ""
	if xticket==0 and rewardsLeft==0:
		ticket=0
	elif xticket==2 and rewardsLeft==120:
		ticket=1
		#checked_text=f"these sellback values haven't been checked yet"
	elif xticket==3 and rewardsLeft==80:
		ticket=2
	elif xticket==2 and rewardsLeft==80:
		ticket=3
	elif xticket==2 and rewardsLeft==240:
		ticket=4
	elif xticket==2 and rewardsLeft==424:
		#korvet
		rewardsLeft=240
		ticket=4
	elif xticket==3 and rewardsLeft==160:
		ticket=5
	else:
		return None, debug_files, f"x{xticket}/{rewardsLeft} chest not coded yet"
	
	prizeList = [prizeA,prizeB,prizeC,prizeD,prizeE,prizeF,prizeG,prizeL]
	totalPulls = sum(prizeList)-prizeL

	totalRev = [a * b for a,b in zip(prizeList,rev[ticket])]
	grandprizeCost = rev[ticket]
	costGetAll = (totalPulls * xticket * 300)
	gemSellAll = sum(totalRev)*10
	ticketSellAll = int(sum(totalRev)/30)

	retval = (
	    f"A : {prizeA}\tB : {prizeB}\tC : {prizeC}\n"+
	    f" \t\tD : {prizeD}\tE : {prizeE}\n"+
	    f"L : {prizeL}\tF : {prizeF}\tG : {prizeG}\n\n"+
	    f"Tickets per pull : x{xticket}\n"+
	    f"Total pulls left : {totalPulls}/{rewardsLeft}\n\n"+
	    f"(tickets/gems)\n"+
	    f"keep all (net): {(-1 * totalPulls*xticket):+d}/{(-1 * costGetAll):+d}\n"+
	    #f"sell all : +{int(sum(totalRev)/30)}/+{sum(totalRev)*10}\n\n"
		f"sell all (net): {(ticketSellAll-(totalPulls * xticket)):+d}/{(gemSellAll-costGetAll):+d}\n\n"
	)
	gemsPerMilestone = abs((gemSellAll-costGetAll)/(totalPulls*xticket))
	milestone_text=f"(milestone)\nsell all : {gemsPerMilestone:.2f} gems/milestone (chest = {int(300*gemsPerMilestone)} gems)\n"
	
	if prizeA==0:
		#retval+=f"sell all (net) : {(sum(totalRev)*10)-costGetAll} gems\n"
		retval+=f"  keep L (net) : {((sum(totalRev)-rev[ticket][0])*10)-costGetAll} gems"
		if ticket in [4,5]:
			milestone_text += f"{int((costGetAll-((sum(totalRev)-(rev[ticket][0]))*10))/(1+(totalPulls*xticket/300)))} gems/chest ({(1+(totalPulls*xticket/300)):.2f})"
		print("prizeA==0")

	elif prizeA==1:
		retval=retval+(
			#f"    sell all (net) : {(sum(totalRev)*10)-costGetAll} gems\n"+
			f"keep 1 A/L (net) : {((sum(totalRev)-rev[ticket][0])*10)-costGetAll} gems\n"+
			f"keep 2 A+L (net) : {((sum(totalRev)-(2*rev[ticket][0]))*10)-costGetAll} gems"
		)
		if ticket in [4,5]:
			gemCost = (costGetAll-((sum(totalRev)-(2*rev[ticket][0]))*10))
			retval=retval+f"\n\n{int((costGetAll-((sum(totalRev)-(2*rev[ticket][0]))*10))/2)} gems/chest (2)"
			milestone_text += f"{int((costGetAll-((sum(totalRev)-(2*rev[ticket][0]))*10))/(2+(totalPulls*xticket/300)))} gems/chest ({(2+(totalPulls*xticket/300)):.2f})"
		print("prizeA==1")

	elif prizeA==2:
		retval=retval+(
			#f"    sell all (net) : {(sum(totalRev)*10)-costGetAll} gems\n"+
			f"keep 1 A/L (net) : {((sum(totalRev)-rev[ticket][0])*10)-costGetAll} gems\n"+
			f"keep 2 A+L (net) : {((sum(totalRev)-(2*rev[ticket][0]))*10)-costGetAll} gems\n"+
			f"keep 3 A+L (net) : {((sum(totalRev)-(3*rev[ticket][0]))*10)-costGetAll} gems"
		)
		if ticket in [4,5]:
			retval=retval+f"\n\n{math.trunc((costGetAll-((sum(totalRev)-(3*rev[ticket][0]))*10))/3)} gems per chest (3)"
			milestone_text += f"{int((costGetAll-((sum(totalRev)-(3*rev[ticket][0]))*10))/(3+(totalPulls*xticket/300)))} gems/chest ({(3+(totalPulls*xticket/300)):.2f})"
		print("prizeA==2")

	retval += f"\n\n{milestone_text}"

	print(retval)
	print(f"error text: {error_text}")

	cv2.imwrite("./debug/cropaniv_appended.png", append_img)
	cv2.imwrite("./debug/cropaniv.png", crop1)
	cv2.imwrite("./debug/topPrice.png", topPrice)
	cv2.imwrite("./debug/botPrice.png", botPrice)
	cv2.imwrite("./debug/rewardsLeft.png", rewardsLeft)
	cv2.imwrite("./debug/debuglive.png", crop1)	
	debug_files.append(discord.File("./debug/debuglive.png", filename = "debuglive.png"))
	debug_files.append(discord.File("./debug/cropaniv_appended.png", filename="cropaniv_appended.png"))

	
	return retval, debug_files, error_text