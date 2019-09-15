import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.color as color
import skimage.filters as filters
import skimage.feature as feature
import skimage.measure as measure
from skimage import segmentation
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import skimage.transform as transform
from skimage import draw

# read in the images
DATA_DIR = 'cw2-data/img/'
img_list = ['rgbh0000.bmp','rgbh0008.bmp','rgbh0014.bmp','rgbh0015.bmp','rgbh0016.bmp']
imm = []
for img in img_list:
    tmp = imageio.imread(DATA_DIR+img)
    imm.append(tmp)

# generate greyscale
grey_im = []
for img in imm:
    tmp = color.rgb2gray(img)
    grey_im.append(tmp)
    plt.imshow(tmp,cmap = plt.cm.gray, interpolation = 'nearest')
    plt.show()

# convert to B&W
bw_im = []
threshold_list = []
for img in grey_im:
    threshold = filters.threshold_otsu(img)
    threshold_list.append(threshold)
    tmp = img>threshold
    bw_im.append(tmp)
    plt.imshow(tmp,cmap = plt.cm.gray, interpolation = 'nearest')
    plt.show()

# detect edge
edge_list = []
for img in grey_im:
    tmp = feature.canny(img,sigma = 3)
    edge_list.append(tmp)
    plt.imshow(tmp,cmap = plt.cm.gray, interpolation = 'nearest')
    plt.show()

# detect contours
for im in imm:
    img = color.rgb2gray( im)

    #-find a good threshold value
    threshold = filters.threshold_otsu( img )
    print 'Otsu method threshold = ', threshold

    #-find contours at the threshold value found above
    contours = measure.find_contours( img, threshold )

    #-plot the results
    fig, (ax0,ax2) = plt.subplots( nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True )

    ax0.imshow( im )
    ax0.axis( 'off' )
    ax0.set_title( 'original image' )

    for n, contour in enumerate( contours ):
        ax2.plot( contour[:,1], contour[:,0], 'k-', linewidth=2 )
    ax2.axis( 'off' )
    ax2.set_title( 'image contours' )

    fig.tight_layout()

    plt.show()

# color detection and label
# detect white area
for image in grey_im:
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(bw, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay,cmap = plt.cm.gray, interpolation = 'nearest')

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 10:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

# detect RGB area

reds = []; greens = []; blues = []
for im in imm:
    reds.append(im[:,:,2])
    greens.append(im[:,:,1])
    blues.append(im[:,:,0])
    
# red areas
c=-1
for image in reds:
    c+=1
    thresh = 15
    bw = closing(image < thresh, square(3))

    cleared = clear_border(bw)


    label_image = label(cleared)
    image_label_overlay = label2rgb(bw, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay,cmap = plt.cm.gray, interpolation = 'nearest')

    for region in regionprops(label_image):
        
        if region.area >= 5:
            
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    
# blue area
c=-1
for image in blues:
    c+=1

    thresh = 15
    bw = closing(image < thresh, square(3))


    cleared = clear_border(bw)


    label_image = label(cleared)
    image_label_overlay = label2rgb(bw, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay,cmap = plt.cm.gray, interpolation = 'nearest')

    for region in regionprops(label_image):
   
        if region.area >= 10:

            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    
# green areas
c=-1
for image in greens:
    c+=1

    thresh = 100
    bw = closing(image < thresh, square(3))

 
    cleared = clear_border(bw)


    label_image = label(cleared)
    image_label_overlay = label2rgb(bw, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay,cmap = plt.cm.gray, interpolation = 'nearest')

    for region in regionprops(label_image):

        if region.area >= 100:
  
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

# detect lines
#-read image from a file into an image object named 'im'
for im in imm:

    #-convert the image to greyscale
    img = color.rgb2gray( im )

    #-perform Canny edge detection
    edges = feature.canny( img )

    #-apply classic straight-line Hough transform
    lines = transform.probabilistic_hough_line( edges, threshold=1, line_length=1, line_gap=3 )

    #-plot the results
    fig, (ax3) = plt.subplots(  figsize=(3, 3), sharex=True, sharey=True )

    for line in lines:
        p0, p1 = line
        ax3.plot(( p0[0], p1[0] ), ( p0[1], p1[1] ))
    ax3.set_xlim(( 0, img.shape[1] ))
    ax3.set_ylim(( img.shape[0], 0 ))
    ax3.set_title( 'Probabilistic Hough' )


    fig.tight_layout()

    plt.show()

# detect circles
image_rgb = imm[0]
image_gray = color.rgb2gray(image_rgb)
edges = feature.canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

# using ellipse transformation
result =transform.hough_ellipse(edges,accuracy = 20,threshold=30,min_size=20, max_size=50)
result.sort(order='accumulator') # sort by accumulator
# get the parameters for ellipse, center and radiuses.

best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# draw the detected ellipse
cy,cx = draw.ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255) 
plt.imshow(image_rgb)
plt.show()
