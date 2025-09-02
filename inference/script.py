#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8

#File: SEGM_icdar_1_0.py
#Version: 1.2
#Version info: changes for Python 3
#Date: 2019-12-30
#Description: Evaluation script that computes Text Segmentation analyzing GT and Detected images at Pixel Level and computing the results.
#Evaluation will be primarily based on the methodology proposed by the organisers in the paper [1], while a typical precision / recall measurement will also be provided for consistency, in the same fashion as [2].
#1. A. Clavelli, D. Karatzas, and J. Llados, "A Framework for the Assessment of Text Extraction Algorithms on Complex Colour Images", in Proceedings of the 9th IAPR Workshop on Document Analysis Systems, Boston, MA, 2010, pp. 19-28.
#2. K. Ntirogiannis, B. Gatos, and I. Pratikakis, "An Objective Methodology for Document Image Binarization Techniques", in Proceedings of the 8th International Workshop on Document Analysis Systems, Nara, Japan, 2008, pp. 217-224


from io import BytesIO
from PIL import Image
import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
import importlib
import re

def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """      
    return {
            'cv2':'cv2',
            'numpy':'np',
            }

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """ 
    return {
                'MINIMAL_PERC' : 0.5,
                'MAXIMAL_PERC' : 0.9,
                'BW_IMAGES':'0'
            }

def validate_data(gtFilePath, submFilePath,evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    #gt_keys = rrc_evaluation_funcs.load_zip_file_keys(gtFilePath,'(?:gt_img|gt_color|gt_skel|gt_bw)_([0-9]+).png')
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,'^gt_bw_([0-9]+)\.png$')
    
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,'res_img_([0-9]+).png',True)
    
    #Validate format of results
    for k in subm:

        if (k in gt) == False :
            raise Exception("The sample %s not present in GT" %k)
        
        input = BytesIO(subm[k])
        im = Image.open(input)
        submImWidth,submImHeight = im.size
        
        if(evaluationParams['BW_IMAGES']=='1'):
            #BW white modes
            if (im.mode != "1" and im.mode != "L" and im.mode != "RGB"):
                raise Exception("Sample image (%s) format is not correct for BW" %k)
        else:
            #Colored images
            if (im.mode != "RGB" and im.mode != "RGBA"):
                raise Exception("Sample image (%s) format is not correct (%s). Must be RGB (or RGBA)." %(k,im.mode) ) 
        
        fd=BytesIO(gt[k])
        im = Image.open(fd)
        imWidth,imHeight = im.size
        if (imWidth != submImWidth or imHeight != submImHeight) :
            raise Exception("Sample image (%s) dimensions (%s,%s) not valid, must be (%s,%s)" %(k,submImWidth,submImHeight,imWidth,imHeight) )  
        
 
def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """
    
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)    
    
    perSampleMetrics = {}
    
    #gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,'^img_[0-9]+_area\.png$','[0-9]+')
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,'^gt_img_([0-9]+)\.txt$')
    gtAreaImages = rrc_evaluation_funcs.load_zip_file(gtFilePath,'^gt_color_([0-9]+)\.png$')
    gtSkelImages = rrc_evaluation_funcs.load_zip_file(gtFilePath,'^gt_skel_([0-9]+)\.png$')
    gtBWImages = rrc_evaluation_funcs.load_zip_file(gtFilePath,'^gt_bw_([0-9]+)\.png$')
    
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,'res_img_([0-9]+)\.png')
   
    px_recallSum = 0
    px_precisionSum = 0
    
    output_items = {}
    
    sum_atoms_gt = 0
    sum_num_atoms_correct = 0
    sum_num_atoms_well = 0
    sum_num_atoms_res = 0
    sum_gtAtoms = 0
    sum_num_well = 0
    sum_num_lost = 0
    sum_num_merged = 0
    sum_num_broken = 0
    sum_num_broken_merged = 0
    sum_num_fp = 0
    
    debug = False
    
    generateImages = True
    
    for resFile in gt:

        #resFile = "6"
        #debug = resFile == "66"
        if debug:
            print("Image " + resFile)
        
        evaluationLog=""
        px_recall = 0
        px_precision = 0
      
        recall = 0
        precision = 0
        hmean = 0
        
        num_fp = 0
        
        gtInfo = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        
        gtInput = BytesIO(gtAreaImages[resFile])
        gtImg = Image.open(gtInput)
        width, height = gtImg.size
        gtImgData = np.array(gtImg)
        
        gtSkelInput = BytesIO(gtSkelImages[resFile])
        gtSkelImg = Image.open(gtSkelInput)

        gtBWInput = BytesIO(gtBWImages[resFile])
        gtBWImg = Image.open(gtBWInput)
        gtBWImgData = np.array(gtBWImg)
        
        matGt = np.zeros( (height, width) )
        #We want only Foreground pixels
        matGt[ (gtBWImgData == 255 ) ] = 1

        #Atoms evaluation
        #Remove Don't care pixels from GT
        #gtImgData.flags.writeable = True
        gtImgData[ np.where(gtBWImgData == 120 ) ] = [255,255,255]     
        
        gtAtoms = get_atoms_from_colored_image(gtImgData,gtInfo,gtSkelImg)
        
        detAtoms = []
        
        num_atoms_correct = 0
        num_atoms_res = 0        
        
        num_lost = 0
        num_broken = 0
        num_merged = 0
        num_broken_merged = 0
        num_well = 0
        
        px_recall = 0
        px_precision = 0
        px_hmean = 0
        
        
        if resFile in subm:
            
            detInput = BytesIO(subm[resFile])
            detImg = Image.open(detInput)
            
            #Pixel evaluation, Part 2/2 *******************************************************
            if(evaluationParams['BW_IMAGES']=='1'):
                detImgBW = detImg.convert("1") #in case that image mode is L
                detBWImgData = np.array(detImgBW)
                #detBWImgData.flags.writeable = True
            else:
                #Image is colored, replace all white pixels to 0 and others to 1
                detImgRGB = detImg.convert("RGB") #in case that image is RGBA
                detImgData = np.array(detImgRGB)
                matDetTemp = np.ones( (height, width,3) )
                matDetTemp[ np.where((detImgData == [255,255,255]).all(axis=2)) ] = [0,0,0]
                detBWImgData = matDetTemp[:,:,0]
            
            #Remove Don't care pixels from Det
            detBWImgData[ (gtBWImgData == 120 ) ] = 0
            
            px_recall,px_precision,px_hmean,pixelLog,resImg = pixel_evaluation(matGt,detBWImgData)
            if generateImages:
                output_items['px_img' + str(resFile) + '.png'] = resImg
            
            evaluationLog += pixelLog
            px_recallSum += px_recall
            px_precisionSum += px_precision
            
            if(evaluationParams['BW_IMAGES']=='1'):
                #Remove Don't care pixels from Det
                detImg = detImg.convert("L") #in case that image mode is 1
                detImgData = np.array(detImg, dtype=np.uint8)
                #detImgData.flags.writeable = True
                detImgData[ gtBWImgData == 120 ] = 0
                #BW image. Each detected component will be a single atom with only one Text Part
                detAtoms = get_atoms_from_bw_image(detImgData)
                
            else:
                detImgRGB = detImg.convert("RGB") #in case that image is RGBA
                detImgData = np.array(detImgRGB)
                #detImgData.flags.writeable = True
                
                #Remove Don't care pixels from Det
                detImgData[ gtBWImgData == 120 ] = [255,255,255]

                #Image is colored, Each distinct color will be a single atom with one or multiple Text Parts
                detAtoms = get_atoms_from_colored_image(detImgData)
                
   
            #Part 1 - label all components of the detection like : "Background", "Whole", "Fraction", "Mixed" or "Fraction & Multiple"
            for atom in detAtoms:
                for textPart in atom['textParts']:
                    #find GT overlapping textParts
                    overlaps = num_overlapping(textPart,gtAtoms)
                    if debug:
                        print("Det TP " + str(textPart['idTp']) + " overlaps withs GT: " + str(len(overlaps)))
                    if (len(overlaps)==0):
                        textPart['label'] = "Background"
                        num_fp+=1
                    elif (len(overlaps)==1):
                        gtTextPart = overlaps[0]
                        maximal = maximal_coverage_area(textPart,overlaps,evaluationParams['MAXIMAL_PERC'],debug)
                        if maximal == False:
                            textPart['label'] = "Mixed"
                        else:
                            minimal = minimal_coverage_area(textPart,gtTextPart,evaluationParams['MINIMAL_PERC'],debug)
                            textPart['label'] = "Whole" if minimal else "Fraction"
                        textPart['idGtAtom'] = gtTextPart['idParentAtom']
                    else:
                        maximal = maximal_coverage_area(textPart,overlaps,evaluationParams['MAXIMAL_PERC'])
                        if maximal == False:
                            textPart['label'] = "Mixed"
                        else:
                            minimal = True
                            for overlap in overlaps:
                                minimal = minimal and minimal_coverage_area(textPart,overlap,evaluationParams['MINIMAL_PERC'],debug)
                            textPart['label'] = "Multiple" if minimal else "Fraction & Mutiple"
            
            #Part 2 - label all components of the gt like : "Well Segmented", "Merged", "Broken", "Broken & Merged" or "Lost"
            for atom in gtAtoms:
                for textPart in atom['textParts']:
                    #find GT overlapping textParts
                    overlaps = num_overlapping(textPart,detAtoms)
                    if (len(overlaps)==0):
                        textPart['label'] = "Lost"
                    else:
                        if debug:
                            print("GT " +  str(atom['idAtom']) + " overlaps=" + str(len(overlaps)))
                        if (len(overlaps)==1):
                            minimal = minimal_coverage_area(overlaps[0],textPart,evaluationParams['MINIMAL_PERC'],debug)
                        else:
                            minimal = minimal_coverage_area_list(overlaps,textPart,evaluationParams['MINIMAL_PERC'],debug)
                            
                        if minimal == False:
                            textPart['label'] = "Lost"
                        else:
                            whole = False
                            multiple = False
                            fraction = False
                            fraction_mult = False
                            for tp in overlaps:
                                whole = whole or tp['label']=="Whole"
                                if tp['label']=="Multiple":
                                    multiple = True
                                elif tp['label']=="Fraction":
                                    fraction = True                                    
                                elif tp['label']=="Fraction & Multiple":
                                    fraction_mult = True
                            if whole:
                                textPart['label'] = "Well Segmented"
                            else:
                                if fraction_mult or (fraction_mult==False and multiple and fraction):
                                    textPart['label'] = "Broken & Merged"
                                else:
                                    if multiple and fraction==False:
                                        textPart['label'] = "Merged"
                                    elif (multiple==False and fraction):
                                        textPart['label'] = "Broken"
                                    else:
                                        textPart['label'] = "Lost"

            #Part 3 - label all atoms of the gt like : "Well Segmented", "Merged", "Broken", "Broken & Merged" or "Lost"

            
            for atom in gtAtoms:
                lost = False
                broken = False
                merged = False
                broken_merged = False                
                for textPart in atom['textParts']:
                    lost = lost or textPart['label']=="Lost"
                    broken = broken or textPart['label']=="Broken"
                    merged = merged or textPart['label']=="Merged"
                    broken_merged = broken_merged or textPart['label']=="Broken & Merged"
                
                if(lost):
                    atom['label'] = "Lost"
                    num_lost += 1
                elif (broken and merged) or broken_merged :
                    atom['label'] = "Broken & Merged"
                    num_broken_merged +=1
                elif broken :
                    atom['label'] = "Broken"
                    num_broken +=1
                elif merged :
                    atom['label'] = "Merged"
                    num_merged +=1
                else :
                    atom['label'] = "Well Segmented"
                    num_well +=1   
                    
                
            #Part 4 - Atom level performance
            if(evaluationParams['BW_IMAGES']=='1'):
                """
                We have to group under the same 'atom' all detected textparts matched with the same atom
                """
                ended = False
                start=0
                cont=0
                
                textPartsLists = []
                for atom in detAtoms:
                    for textPart in atom['textParts']:
                        textPartsLists.append(textPart)
                
                #[atom['textParts'] for atom in detAtoms]
                maximum = len(textPartsLists)
                
                while (ended == False and start<len(textPartsLists) and cont<maximum  ):
                    textPart = textPartsLists[start]
                    if int(textPart['idGtAtom'])>-1:
                        parentAtom = get_atom_by_id(textPart['idParentAtom'],detAtoms)
                        for j in range(start+1,len(textPartsLists)):
                            textPart2 = textPartsLists[j]
                            if textPart['idGtAtom'] == textPart2['idGtAtom']:
                                parentAtom2 = get_atom_by_id(textPart2['idParentAtom'],detAtoms)
                                parentAtom2['textParts'].remove(textPart2)
                                
                                if len(parentAtom2['textParts'])==0:
                                    detAtoms.remove(parentAtom2)
                                
                                textPart2['idParentAtom'] = textPart['idParentAtom']
                                parentAtom['textParts'].append(textPart2)
                                break
                    start += 1
                    cont += 1
                    textPartsLists = []
                    for atom in detAtoms:
                        for textPart in atom['textParts']:
                            textPartsLists.append(textPart)
                            
            #Final

            
            
            for atom in detAtoms:
                num_atoms_res += 1

                whole = False
                multiple = False
                fraction = False
                fraction_mult = False
                false_pos = True
                    
                for textPart in atom['textParts']:
                    whole = whole or textPart['label'] == "Whole"
                    false_pos = false_pos and textPart['label'] == "Background"
                    if(textPart['label'] == "Multiple"):
                        multiple = True
                    elif (textPart['label'] == "Fraction"):
                        fraction = True
                    elif (textPart['label'] == "Fraction & Multiple"):
                        fraction_mult = True
                
                if whole :
                    atom['label'] = "Well Segmented"
                else:
                    if fraction_mult or ( fraction_mult==False and multiple and fraction ):
                        atom['label'] = "Broken & Merged"
                    else:
                        if multiple and fraction==False:
                            atom['label'] = "Merged"
                        elif multiple==False and fraction:
                            atom['label'] = "Broken"
                        else:
                            atom['label'] = "Lost"
                            
                
                #To be considered correct, all text parts of the detected atom must be 'Whole', refer to the same GT atom , and that all
                correct = False
                ids_gt_atoms_list = list(set([ textPart['idGtAtom'] for textPart in atom['textParts'] ]))
                det_atoms_type_list = list(set([ textPart['label'] for textPart in atom['textParts'] ]))
                if len(ids_gt_atoms_list)==1 and len(det_atoms_type_list)==1:
                    correct = det_atoms_type_list[0]=="Whole" and len(atom['textParts']) >= get_text_parts_length_of_atom_by_id(ids_gt_atoms_list[0],gtAtoms)

                if correct:
                    num_atoms_correct += 1
            
        if generateImages:
            imgAtomsDet,imgAtomsGT = generate_atom_images(gtAtoms,detAtoms,width,height)
            output_items['res_atoms_det_' + resFile + '.png'] = imgAtomsDet
            output_items['res_atoms_gt_' + resFile + '.png'] = imgAtomsGT
        
        
        recall = float(num_atoms_correct) / float(len(gtAtoms));
        precision = 0 if num_atoms_res==0 else float(num_atoms_correct) / float(num_atoms_res);
        hmean = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall);
        
        num_tp = 0
        for atom in gtAtoms:
            num_tp += len(atom['textParts'])
        num_tp_det = 0
        for atom in detAtoms:
            num_tp_det += len(atom['textParts'])
        
        evaluationLog += "num_atoms_correct = " + str(num_atoms_correct) + "\n"
        evaluationLog += "num_atoms_res = " + str(num_atoms_res) + "\n"
        evaluationLog += "num_atoms_gt = " + str(len(gtAtoms)) + "\n"
        evaluationLog += "num_tp_gt = " + str(num_tp) + "\n"
        evaluationLog += "Well = " + str(num_well) + "\n"
        evaluationLog += "Lost = " + str(num_lost) + "\n"
        evaluationLog += "Merged = " + str(num_merged) + "\n"
        evaluationLog += "Broken = " + str(num_broken) + "\n"
        evaluationLog += "BrokenMerged = " + str(num_broken_merged) + "\n"
        evaluationLog += "FalseP = " + str(num_fp) + "\n"

        perSampleMetrics[resFile] = {
                                        'px_precision':px_precision,
                                        'px_recall':px_recall,
                                        'px_fscore':px_hmean,        
                                        'precision':precision,
                                        'recall':recall,
                                        'fscore':hmean,
                                        'num_atoms_correct':num_atoms_correct,
                                        'num_atoms_det':num_atoms_res,
                                        'num_atoms_gt':len(gtAtoms),
                                        'well':num_well,
                                        'lost':num_lost,
                                        'merged':num_merged,
                                        'broken':num_broken,
                                        'brok_merged':num_broken_merged,
                                        'num_fp':num_fp,
                                        'evaluationParams': evaluationParams,
                                        'evaluationLog': evaluationLog     
                                    }
                                    
        sum_atoms_gt += len(gtAtoms)
        sum_num_atoms_correct += num_atoms_correct
        sum_num_atoms_well += num_well
        sum_num_atoms_res += num_atoms_res
        sum_gtAtoms += len(gtAtoms)
        sum_num_well += num_well
        sum_num_lost += num_lost
        sum_num_merged += num_merged
        sum_num_broken += num_broken
        sum_num_broken_merged += num_broken_merged
        sum_num_fp += num_fp
        
    px_methodRecall = float(px_recallSum)/len(gt)
    px_methodPrecision = float(px_precisionSum)/len(gt)
    px_methodHmean = 0 if px_methodRecall + px_methodPrecision==0 else 2* px_methodRecall * px_methodPrecision / (px_methodRecall + px_methodPrecision)    


    methodRecall = float(sum_num_atoms_well)/sum_atoms_gt
    methodPrecision = 0 if sum_num_atoms_res==0 else float(sum_num_atoms_correct)/float(sum_num_atoms_res)
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)
    
    methodMetrics = {
                        'px_precision':px_methodPrecision, 
                        'px_recall':px_methodRecall,
                        'px_fscore': px_methodHmean,
                        'precision':methodPrecision,
                        'recall':methodRecall,
                        'fscore': methodHmean,
                        'num_atoms_correct':sum_num_atoms_correct,
                        'num_atoms_det':sum_num_atoms_res,
                        'num_atoms_gt':sum_gtAtoms,
                        'num_tp':num_tp,
                        'num_tp_det':num_tp_det,
                        'well':sum_num_well,
                        'lost':sum_num_lost,
                        'merged':sum_num_merged,
                        'broken':sum_num_broken,
                        'brok_merged':sum_num_broken_merged,
                        'num_fp':sum_num_fp
                    }

    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics,'output_items':output_items}
    
    
    return resDict;


def distinct_colors(img):
    """Return the distinct colors found in img.
    img must be an Image object (from the Python Imaging Library).
    The result is a Numpy array with three columns containing the
    red, green and blue values for each distinct color.

    """
    width, height = img.size
    colors = [rgb for _, rgb in img.getcolors(width * height)]
    return np.array(colors, dtype=np.uint8)

def hex_to_int_color(v):
    if v[0] == '#':
        v = v[1:]
    assert(len(v) == 6)
    return [int(v[:2], 16), int(v[2:4], 16), int(v[4:6], 16)]   

def distinct_colors_from_gt_txt(txt):
    colors = []
    lines = txt.split( '\r\n' )
    for line in lines:
        if line != "":
            m = re.match(r'^(#[0-9|a-f]{6}),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+),(.*)$',line)
            if m == None :
                raise Exception("GT incorrect. (" + line + ")")

            color = hex_to_int_color(m.group(1))
            colors.append( [color ,[int(m.group(2)),int(m.group(3)),int(m.group(4)),int(m.group(5)),int(m.group(6)),m.group(7)]])
    return colors


def get_atoms_from_colored_image(imgData,gtInfo=None,imgSkel=None):
    """
    Extracts components from a colored image. Each color defines one atom that can have multiple Text Parts.
    Parameters:
    imgData: The Numpy array of the colored image
    gtInfo: (optional) The GroundThuth for that image. It's used to improve performance by reading the boundaries 
    of each atom and detecting components only inside that regions.
    imgSkel: (required for GT) The Numpy array of the skeleton image. Each skel color must match the atom color.
    """

    width = imgData.shape[1]
    height = imgData.shape[0]

    idAtom=0
    idTp=0

    detAtoms = []

    im = Image.fromarray(imgData.astype('uint8'))

    #Detect distinct colours of the image. Each color will be one atom
    if gtInfo==None:
        colors = distinct_colors(im)
    else:
        tuplesColorInfo = distinct_colors_from_gt_txt(gtInfo)

        colorsList = [color for color,_ in tuplesColorInfo]
        colors = np.array(colorsList, dtype=np.uint8)

        colorsInfo = [info for _,info in tuplesColorInfo]

    #Boundaries where apply the found components function
    xmin = ymin = 0
    xmax = int(width)-1
    ymax= int(height)-1



    for k in range(0,len(colors)):

        atomTranscription=""
        color = colors[k]
        if gtInfo != None:
            [xmin,xmax,ymin,ymax,numTextParts,atomTranscription] = colorsInfo[k]

        #exclude background (white in colored images)
        if (color.tolist() != [255,255,255]):
            atWidth = xmax-xmin+1
            atHeight = ymax-ymin+1
            atDetTemp = np.zeros( (atHeight, atWidth,3), np.uint8 )
            atDetTemp[ np.where((imgData[ymin:ymax+1,xmin:xmax+1] == color).all(axis=2)) ] = [255,255,255]
            atDetTemp2 = np.zeros( (atHeight, atWidth), np.uint8 )
            atDetTemp2[:,:] = atDetTemp[:,:,0]

            if imgSkel != None:
                #convert to BW the skeleton Image and filter with only the skeletons of the current atom (color)
                skelImgData = np.asarray(imgSkel)
                tpSkelTemp = np.zeros( (atHeight, atWidth,3), np.uint8 )
                tpSkelTemp[ np.where((skelImgData[ymin:ymax+1,xmin:xmax+1] == color).all(axis=2)) ] = [255,255,255]

            detectComponents = True
            if gtInfo != None:
                detectComponents = numTextParts>1

            if detectComponents:
                num_labels,labels,stats,_ = cv2.connectedComponentsWithStats(atDetTemp2, 8, cv2.CV_32S)

                #if num_labels>2:
                #    imTP_GT = Image.fromarray(atDetTemp2.astype('uint8'))
                #    imTP_GT.save(resultsFolder + '/atoms_mult_tp_' + str(idAtom) + '.png')                       
                #    print "idAtom " + str(idAtom) + " textparts = " + str(num_labels-1)

                textParts = []

                for k in range(1,num_labels):
                    matTP = np.zeros( (atHeight, atWidth), np.uint8 ) 
                    matTP[ labels == k] = 255
                    #Create Text part Image data of image size
                    matTpImg = np.zeros( (height, width), np.uint8 ) 
                    matTpImg[ymin:ymax+1,xmin:xmax+1]  = matTP

                    matSkelImg = None
                    if imgSkel != None:

                        #For compatibility, not exclude pixels outside the area
                        matSkel = matTP * tpSkelTemp[:,:,0]
                        matSkel[matSkel>0]=255

                        #matSkel = tpSkelTemp[:,:,0]

                        #Create Skel Image data of image size
                        matSkelImg = np.zeros( (height, width), np.uint8 ) 
                        matSkelImg[ymin:ymax+1,xmin:xmax+1]  = matSkel

                    coords = stats[k]
                    coords[0] += xmin
                    coords[1] += ymin

                    textParts.append( {"idTp":idTp,"num":k,"bwImg":matTpImg,"label":"","coords":stats[k],"idParentAtom":idAtom,"skelImg":matSkelImg,"idGtAtom":-1} )
                    idTp += 1
            else:
                #We don't have to detect components, add all pixels to the same TextPart

                atAreaImg = np.zeros( (height, width), np.uint8 ) 
                atAreaImg[ymin:ymax+1,xmin:xmax+1]  = atDetTemp2    

                matSkelImg = None
                if imgSkel != None:
                    matSkelImg = np.zeros( (height, width), np.uint8 ) 
                    matSkelImg[ymin:ymax+1,xmin:xmax+1]  = tpSkelTemp[:,:,0]

                coords = [xmin, ymin, xmax-xmin+1, ymax-ymin+1, 0 ]

                textParts = [{"idTp":idTp,"num":1,"bwImg":atAreaImg,"label":"","coords":coords,"idParentAtom":idAtom,"skelImg":matSkelImg,"idGtAtom":-1}]
                idTp += 1
                #print textParts

            atom = {"idAtom":idAtom,"color":color,"bwImg":atDetTemp2,"textParts":textParts,"label":"","transcription":atomTranscription}

            detAtoms.append(atom)

        idAtom += 1

    return detAtoms

def get_atoms_from_bw_image(imgData):

    width = imgData.shape[1]
    height = imgData.shape[0]

    idAtom=0
    idTp=0

    detAtoms = []

    num_labels,labels,stats,_ = cv2.connectedComponentsWithStats(imgData, 8, cv2.CV_32S)

    for k in range(1,num_labels):

        matTP = np.zeros( (height, width), np.uint8 ) 
        matTP[ labels == k] = 255

        textParts = []
        textParts.append( {"idTp":idTp,"num":k,"bwImg":matTP,"label":"","idParentAtom":idAtom,"coords":stats[k],"idGtAtom":-1} )
        atom = {"idAtom":idAtom,"bwImg":matTP,"textParts":textParts,"label":""}
        detAtoms.append(atom)

        idAtom += 1
        idTp += 1

    return detAtoms

def num_overlapping(textPart,atomsList):
    overlaps = []
    for atom in atomsList:
        for gtTextPart in atom['textParts']:
            overlap,[xmin,xmax,ymin,ymax] = textPartsRectangleOverlap(textPart,gtTextPart)
            if overlap==True:
                #The bounding boxes overlaps, see if the area overlaps
                res = textPart['bwImg'][ymin:ymax+1,xmin:xmax+1] & gtTextPart['bwImg'][ymin:ymax+1,xmin:xmax+1]
                if np.count_nonzero(res)>0:
                    overlaps.append(gtTextPart)

    return overlaps

def calculate_thickness(imgArea,imgSkel,debug):
    cnts, _ = cv2.findContours(imgArea, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #if (len(cnts)!=1):
    #    print " Number of TextPart perimeters incorrect: " + str(len(cnts))
    #    return 1

    cnt = cnts[0]
    minx = np.min(cnt[:,0,0])
    miny = np.min(cnt[:,0,1])
    maxx = np.max(cnt[:,0,0])
    maxy = np.max(cnt[:,0,1])

    for cnt in cnts:
        minx = min(minx,np.min(cnt[:,0,0]))
        miny = min(miny,np.min(cnt[:,0,1]))
        maxx = max(maxx,np.max(cnt[:,0,0]))
        maxy = max(maxy,np.max(cnt[:,0,1]))

    tpW = maxx-minx+1
    tpH = maxy-miny+1

    #We need to add extra padding for the dilatation
    matTP = np.zeros( (tpH*3,tpW*3) );
    matSkel = np.zeros( (tpH*3,tpW*3) );
    paddX = tpW
    paddY = tpH

    for cnt in cnts:
        cnt[:,0,0]-=minx
        cnt[:,0,1]-=miny
        for point in cnt:
            matTP[point[0][1]+paddY][point[0][0]+paddX]=255

    matSkel[paddY:tpH+paddY,paddX:tpW+paddX] = imgSkel[miny:maxy+1,minx:maxx+1]
    matSkel[matSkel>0]=255

    numSkelPoints = np.count_nonzero(matSkel)

    if(numSkelPoints==0):
        if debug:
            print("Error ON calculate_thickness, No Skel points!")
        return -1

    perimeterPoints = np.count_nonzero(matTP)

    resting_perimeter_points = perimeterPoints

    level=0
    total_distance = 0


    kernel = np.ones((3,3),np.uint8)
    matSkel = cv2.dilate(matSkel,kernel,iterations = 1)

    while (resting_perimeter_points>0):

        matTP = matTP - matSkel
        matTP[matTP<0]=0

        numPointsInLevel = resting_perimeter_points-np.count_nonzero(matTP)

        total_distance += numPointsInLevel * level

        level +=1

        resting_perimeter_points = np.count_nonzero(matTP)

        matSkel = cv2.dilate(matSkel,kernel,iterations = 1)

        if(level>tpH and level>tpW and level>1000):
            if debug:
                print("Error ON calculate_thickness Level=" + str(level) + " w=" + str(tpW) + " h=" + str(tpH))
            return -1

    return round( float(total_distance) / float(perimeterPoints) )


def textPartsRectangleOverlap(textPart1,textPart2):
    """
    Determines if 2 the bounding boxes of 2 textParts overlaps and returns the Bounding box of the 2 textparts
    """
    x1,y1,w1,h1,_ = textPart1['coords']
    x2,y2,w2,h2,_ = textPart2['coords']

    xmin = min(x1,x2)
    xmax = max(x1+w1,x2+w2)
    ymin = min(y1,y2)
    ymax = max(y1+h1,y2+h2)

    if (x1+w1<x2 or x2+w2<x1 or y1+h1<y2 or y2+h2<y1):
        return False, [xmin , xmax , ymin, ymax ]

    return True, [xmin , xmax , ymin, ymax ]

def minimal_coverage_area(textPartDet,textPartGT,MINIMAL_PERC,debug=False):

    #Detect if the TextParts overlaps
    x1,y1,w1,h1,_ = textPartDet['coords']
    x2,y2,w2,h2,_ = textPartGT['coords']
    if (x1+w1<x2 or x2+w2<x1 or y1+h1<y2 or y2+h2<y1):
        percentage=0
    else:
        #If the TextParts overlaps, calculate the percentage of GT detected
        xmin = min(x1,x2)
        xmax = max(x1+w1,x2+w2)
        ymin = min(y1,y2)
        ymax = max(y1+h1,y2+h2)

        res = textPartDet['bwImg'][ymin:ymax+1,xmin:xmax+1] & textPartGT['bwImg'][ymin:ymax+1,xmin:xmax+1]

        pixelsDetexted = np.count_nonzero(res)
        pixelsInGT = np.count_nonzero(textPartGT['bwImg'][ymin:ymax+1,xmin:xmax+1] )
        percentage = float(pixelsDetexted) / float(pixelsInGT)
    if debug:
        #atom = get_atom_by_id(textPartGT['idParentAtom'],gtAtoms) (" + atom['transcription'] + ")
        print("minimal_coverage_area between TP Det #" + str(textPartDet['idTp']) + " and TP GT #" + str(textPartGT['idTp']) + " Atom:" + str(textPartGT['idParentAtom']) + " = " + str(percentage))

    return percentage >= MINIMAL_PERC

def minimal_coverage_area_list(textPartDetList,textPartGT,MINIMAL_PERC,debug=False):

    x,y,w,h,_ = textPartGT['coords']
    xmin = x
    ymin = y
    xmax = x+w-1
    ymax = y+h-1

    for textPart in textPartDetList:
        x1,y1,w1,h1,_ = textPart['coords']
        xmin = min(xmin,x1)
        ymin = min(ymin,y1)
        xmax = max(xmax,x1+w1-1)
        ymax = max(ymax,y1+h1-1)

    sumImg = np.zeros( (ymax-ymin+1, xmax-xmin+1) )
    for textPartDet in textPartDetList:
        sumImg += textPartDet['bwImg'][ymin:ymax+1,xmin:xmax+1]

    res = sumImg * textPartGT['bwImg'][ymin:ymax+1,xmin:xmax+1]

    pixelsDetected = np.count_nonzero(res)
    pixelsInGT = np.count_nonzero(textPartGT['bwImg'][ymin:ymax+1,xmin:xmax+1])
    percentage = float(pixelsDetected) / float(pixelsInGT)

    if debug:
        print("minimal_coverage_area_list percentage=" + str(percentage))

    return percentage >= MINIMAL_PERC    

def maximal_coverage_area(textPartDet,textPartGTList,MAXIMAL_PERC,debug=False):

    x,y,w,h,_ = textPartDet['coords']
    xmin = x
    ymin = y
    xmax = x+w-1
    ymax = y+h-1

    for textPart in textPartGTList:
        x,y,w,h,_ = textPart['coords']

        xmin = min(xmin,x)
        ymin = min(ymin,y)
        xmax = max(xmax,x+w-1)
        ymax = max(ymax,y+h-1)

    textPartDetImg = textPartDet['bwImg'][ymin:ymax+1,xmin:xmax+1]

    sumImg = np.zeros( ((ymax-ymin+1, xmax-xmin+1)) )
    for textPartGT in textPartGTList:

        img = textPartGT['bwImg'][ymin:ymax+1,xmin:xmax+1]
        imgSkel = textPartGT['skelImg'][ymin:ymax+1,xmin:xmax+1]

        thickness = calculate_thickness(img,imgSkel,debug) * 2
        if (thickness<=0):
            thickness = 1
        times = int(round(MAXIMAL_PERC * float(thickness) * 2,0))

        if debug:
            print("dilate GT TP #" + str(textPartGT['idTp']) + " " + str(times)  + " times")

        #dilate the image N times
        kernel = np.ones((3,3),np.uint8)

        if debug and textPartDet['idTp']==0:
            imTP = Image.fromarray(img.astype('uint8'))
            imTP.save(resultsFolder + '/before' + str(textPartGT['idTp']) + '.png')   
            dilation = img

            for k in range(1,times):
                dilation = cv2.dilate(dilation,kernel,iterations = 1)
                imTP = Image.fromarray(dilation.astype('uint8'))
                imTP.save(resultsFolder + '/iter' + str(textPartGT['idTp']) + '-' + str(k) + '.png')   

        else:
            dilation = cv2.dilate(img,kernel,iterations = times)

        if debug and textPartDet['idTp']==0:
            imTP = Image.fromarray(dilation.astype('uint8'))
            imTP.save(resultsFolder + '/after_dil_' + str(textPartGT['idTp']) + '.png')            

        sumImg += dilation

    res = textPartDetImg * sumImg
    pixelsDetexted = np.count_nonzero(res)
    pixelsTotal = np.count_nonzero(textPartDetImg)
    percentage = float(pixelsDetexted) / float(pixelsTotal)


    if debug and textPartDet['idTp']==0:
        imTP = Image.fromarray(textPartDetImg.astype('uint8'))
        imTP.save(resultsFolder + '/textPartDetImg.png')      
        print("pixelsDetexted =" + str(pixelsDetexted) + " pixelsTotal = " + str(pixelsTotal))
        print("maximal_coverage_area between TP Det #" + str(textPartDet['idTp']) + " and GT #" + str([textPart['idTp'] for textPart in textPartGTList ])  + " = " + str(percentage))

    return percentage >= 1.0

def get_text_parts_length_of_atom_by_id(idAtom,atomsList):
    for atom in atomsList:
        if atom["idAtom"]==idAtom:
            return len(atom['textParts'])
    return 0

def get_atom_by_id(idAtom,atomsList):
    for atom in atomsList:
        if atom["idAtom"]==idAtom:
            return atom
    return None


def pixel_evaluation(gtBWImgData,detImgBW):
    
    width = int(gtBWImgData.shape[1])
    height = int(gtBWImgData.shape[0])
    
    #BW GT img pixels are 0: background, 120: don't care, 255:Foreground
    num_gt_bg_px = np.count_nonzero(gtBWImgData==0)
    num_gt_fg_px = np.count_nonzero(gtBWImgData==1)
    #num_gt_dc_px = np.count_nonzero(gtBWImgData==120)

    evalLog = "Image size: " + str(width) + " x " + str(height) + "\n"
    evalLog += "GT Background pixels: " + str(num_gt_bg_px) + "\n"
    evalLog += "GT Foreground pixels: " + str(num_gt_fg_px) + "\n"
    #evalLog += "GT Don't care pixels: " + str(num_gt_dc_px) + "\n"        

    numDETbackGroundPx = np.count_nonzero(detImgBW==0)
    numDETforeGroundPx = np.count_nonzero(detImgBW==1)
    matCorrect = gtBWImgData * detImgBW

    matOthers = detImgBW - gtBWImgData

    numDETforeGroundPxAfter = np.count_nonzero(detImgBW==1)
    num_ko = np.count_nonzero(matOthers==1)
    num_no_det = np.count_nonzero(matOthers==-1)
    num_correct = np.count_nonzero(matCorrect)

    px_recall = 0 if num_gt_fg_px==0 else float(num_correct) / float(num_gt_fg_px)
    px_precision = 0 if numDETforeGroundPxAfter==0 else float(num_correct) / float(numDETforeGroundPxAfter)
    px_hmean = 0 if (px_precision + px_recall)==0 else 2.0 * px_precision * px_recall / (px_precision + px_recall)
    
    #generate the result image
    matResult =  np.array(matOthers)
    matResult[ (matResult == -1) ] = 2
    matResult[ (matResult == 1) ] = 3
    matResult += matCorrect

    imResult = np.zeros((height, width,3))
    imResult[:,:,0] = matResult[:,:]
    imResult[:,:,1] = matResult[:,:]
    imResult[:,:,2] = matResult[:,:]

    imResult[ np.where((imResult == [1,1,1]).all(axis=2)) ] = [0,90,0]
    imResult[ np.where((imResult == [2,2,2]).all(axis=2)) ] = [255,255,255]
    imResult[ np.where((imResult == [3,3,3]).all(axis=2)) ] = [255,0,0]

    im = Image.fromarray(imResult.astype('uint8'))
    output = BytesIO()
    im.save(output,'PNG')
    imgResPixel = output.getvalue()
    output.close()

    num_total_px = width * height
    evalLog += "DET Background pixels: " + str(numDETbackGroundPx) + "\n"
    evalLog += "DET Foreground pixels: " + str(numDETforeGroundPx) + "\n"
    evalLog += "DET Don't care pixels: " + str(numDETforeGroundPx-numDETforeGroundPxAfter) + "\n"
    evalLog += "Correct pixels: " + str(num_correct) + " " + str(round(float(num_correct)/num_total_px*100,1)) + "%\n"
    evalLog += "Incorrect pixels: " + str(num_ko) + " " + str(round(float(num_ko)/num_total_px*100,1)) + "%\n"
    evalLog += "No detected pixels: " + str(num_no_det) + " " + str(round(float(num_no_det)/num_total_px*100,1)) + "%\n"

    return px_recall,px_precision,px_hmean,evalLog,imgResPixel

def generate_atom_images(gtAtoms,detAtoms,w,h):
    
    arrImFinalDet = np.zeros( (h,w,3), np.uint8 );
    arrImFinalGt = np.zeros( (h,w,3), np.uint8 );

    for atom in gtAtoms:
        for tp in atom['textParts']:
            if atom['label']=="Well Segmented":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=255
                arrImFinalGt[:,:,1] += tp['bwImg']
            elif atom['label']=="Merged":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=38
                arrImFinalGt[:,:,0] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=148
                arrImFinalGt[:,:,1] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=232
                arrImFinalGt[:,:,2] += tp['bwImg']
            elif atom['label']=="Broken":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=247
                arrImFinalGt[:,:,0] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=149
                arrImFinalGt[:,:,1] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=34
                arrImFinalGt[:,:,2] += tp['bwImg']  
            elif atom['label']=="Broken & Merged":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=50
                arrImFinalGt[:,:,0] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=20
                arrImFinalGt[:,:,1] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=0
                arrImFinalGt[:,:,2] += tp['bwImg']                      
            elif atom['label']=="Lost":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=255
                arrImFinalGt[:,:,0] += tp['bwImg']            
    #    print "GT Atom id=" + str(atom['idAtom']) + " color=" + str(atom['color']) + " textparts=" + str(len(atom['textParts'])) + " label=" + atom['label']   
    #    imTP = Image.fromarray(atom['bwImg'].astype('uint8'))
    #    imTP.save(resultsFolder + '/at_gt_' + str(atom['idAtom']) + '.png')                         
    #    for tp in atom['textParts']:
    #        print "TextPart id=" + str(tp['idTp']) + " label=" + tp['label']   

    for atom in detAtoms:
        #print " Atom id=" + str(atom['idAtom']) + " textparts=" + str(len(atom['textParts'])) + " label=" + atom['label']
        #imTP = Image.fromarray(atom['bwImg'].astype('uint8'))
        #imTP.save(resultsFolder + '/at_detected_' + str(atom['idAtom']) + '.png')                         
        for tp in atom['textParts']:
            if atom['label']=="Well Segmented":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=255
                arrImFinalDet[:,:,1] += tp['bwImg']
            elif atom['label']=="Merged":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=38
                arrImFinalDet[:,:,0] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=148
                arrImFinalDet[:,:,1] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=232
                arrImFinalDet[:,:,2] += tp['bwImg']
            elif atom['label']=="Broken":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=247
                arrImFinalDet[:,:,0] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=149
                arrImFinalDet[:,:,1] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=34
                arrImFinalDet[:,:,2] += tp['bwImg']      
            elif atom['label']=="Broken & Merged":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=50
                arrImFinalDet[:,:,0] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=20
                arrImFinalDet[:,:,1] += tp['bwImg']
                tp['bwImg'][tp['bwImg']>0]=0
                arrImFinalDet[:,:,2] += tp['bwImg']
            elif atom['label']=="Lost":
                imTP = Image.fromarray(tp['bwImg'].astype('uint8'))
                tp['bwImg'][tp['bwImg']>0]=255
                arrImFinalDet[:,:,0] += tp['bwImg']

                #print "TextPart id=" + str(tp['idTp']) + " (atom " + str(tp['idParentAtom']) + ") label=" + tp['label'] + "gt=" + str(tp['idGtAtom'])

    imTP = Image.fromarray(arrImFinalDet.astype('uint8'))
    output = BytesIO()
    imTP.save(output,'PNG')
    imgAtomsDet = output.getvalue()
    output.close()

    output2 = BytesIO()
    imTP_GT = Image.fromarray(arrImFinalGt.astype('uint8'))
    imTP_GT.save(output2,'PNG')
    imgAtomsGT = output2.getvalue()
    output2.close()
    
    return imgAtomsDet,imgAtomsGT

if __name__=='__main__':
        
    rrc_evaluation_funcs.main_evaluation(None,default_evaluation_params,validate_data,evaluate_method)