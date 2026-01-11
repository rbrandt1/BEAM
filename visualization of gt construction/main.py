import cv2
import numpy as np
import random


settings_global = {}


def saveImage(path, image):
    cv2.imwrite(path, image*255) 


def saveImageColomap(path, image):

    image = convert_to_2D(image)
    
    im_color = (((image+1)/2)*255).astype(np.uint8)
    cv2.imwrite(path, im_color)



def normalize_explanation(exp):
   
    max = np.amax(exp)
    min = np.amin(exp)
    
    if max != 0 or min != 0:
        if abs(min) > abs(max):
            exp = exp * 1.0/abs(min)
        else:
            exp = exp * 1.0/abs(max)
    
    if (np.amin(exp) < -1 or np.amax(exp) > 1) or ((np.amin(exp) != -1 and np.amax(exp) != 1) and not (np.amin(exp) == 0 and np.amax(exp) == 0)):
        print("ERROR: np.amin(exp) < -1 or np.amax(exp) > 1 ")
        print(np.amin(exp),np.amax(exp))
        exit()

    return exp
    
def gen_concept_part_weights(id):

    if id == 0:
        return [[2,-.5,1],[-.5,0,-1],[1,0,-1]] # row 2 is residual,     0,2       1,2
    if id == 1:
        return [[0,1,1],[1,-1,-1],[1,0,-1]] # row 2 is residual,     0,2       1,2
    if id == 2:
        return [[0,0,1],[0,2,-1],[1,0,-1]] # row 2 is residual,     0,2       1,2
    if id == 3:
        return [[1,-.5,2],[-1,0,-.5],[1,0,-1]] # row 2 is residual,     0,0       1,0
    if id == 4:
        return [[1,1,0],[-1,-1,1],[1,0,-1]] # row 2 is residual,     0,0       1,0
    if id == 5:
        return [[1,0,0],[-1,2,0],[1,0,-1]] # row 2 is residual,     0,0       1,0
    if id == 6:
        return [[1,0,-1],[1,0,-.5],[-1,-.5,2]] # row 0 is residual,     1,0       2,0
    if id == 7:
        return [[1,0,-1],[1,-1,1],[-1,1,0]]  # row 0 is residual,     1,0       2,0
    if id == 8:
        return [[1,0,-1],[1,2,0],[-1,0,0]]  # row 0 is residual,     1,0       2,0
    if id == 9: 
        return [[1,0,-1],[-.5,0,1],[2,-.5,-1]]  # row 0 is residual,     1,2       2,2
    if id == 10:
        return [[1,0,-1],[1,-1,1],[0,1,-1]]  # row 0 is residual,     1,2       2,2
    if id == 11:
        return [[1,0,-1],[0,2,1],[0,0,-1]] # row 0 is residual,     1,2       2,2
        
def gen_concept_part_input_exp(id):
    if id == 0:
        a =  [[1,0,0],[0,0,0],[.5,1,.5]]
    if id == 1:
        a =  [[1,1,0],[1,0,0],[.5,1,.5]]
    if id == 2:
        a =  [[1,1,0],[1,1,0],[.5,1,.5]]
    if id == 3:
        a =  [[0,0,1],[0,0,0],[.5,1,.5]]
    if id == 4:
        a =  [[0,1,1],[0,0,1],[.5,1,.5]]
    if id == 5:
        a =  [[0,1,1],[0,1,1],[.5,1,.5]]
    if id == 6:
        a =  [[.5,1,.5],[0,0,0],[0,0,1]]
    if id == 7:
        a =  [[.5,1,.5],[0,0,1],[0,1,1]]
    if id == 8:
        a =  [[.5,1,.5],[0,1,1],[0,1,1]]
    if id == 9:
        a =  [[.5,1,.5],[0,0,0],[1,0,0]]
    if id == 10:
        a =  [[.5,1,.5],[1,0,0],[1,1,0]]
    if id == 11:
        a =  [[.5,1,.5],[1,1,0],[1,1,0]]

    
    b = gen_concept_part_weights(id)

    return np.array(a), np.array(b)
        

def gen_concept_part_example(id):
    
    inputs, exps = gen_concept_part_input_exp(id)

    return inputs, exps



     
def calc_attr_neg_cp(img_0, exp_1, multiplier, enabled):
    
    if settings_global['stage'] >= 3:

        if not enabled:
            assert False

        img_0 = np.array(img_0)
        exp_1 = np.array(exp_1)
        
        total_exp_concept = abs(exp_1).sum()
        if total_exp_concept == 0:
            total_exp_concept = 1

        tmp = np.zeros(img_0.shape)

        for y in range(img_0.shape[0]):
            for x in range(img_0.shape[1]):
                if len(img_0.shape) == 3:
                    for c in range(img_0.shape[2]):
                        tmp[y,x,c] = ((- img_0[y,x,c] * exp_1[y,x,c]) + 2 * abs(exp_1[y,x,c])/total_exp_concept) 
                else:
                    tmp[y,x] = ((- img_0[y,x] * exp_1[y,x]) + 2 * abs(exp_1[y,x])/total_exp_concept) 
      
        if np.abs(np.sum(tmp)) > .0000000000001:
            tmp = tmp / np.sum(tmp)
        tmp = tmp * multiplier

    else:
        tmp = np.ones_like(img_0) * multiplier

    return img_0, tmp
    

    
def calc_attr_pos_cp(array, multiplier, enabled = True):

    img,exp = array

    if settings_global['stage'] >= 3:

        if enabled:
            total_exp_concept = abs(exp).sum()
            if total_exp_concept == 0:
                total_exp_concept = 1

            tmp = np.zeros(img.shape)

            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    tmp[y,x] =  ((img[y,x] * exp[y,x])  -1 * abs(exp[y,x])/total_exp_concept )

            tmp = (tmp / (tmp.sum())) * multiplier

            exp = tmp
            
    else:
        exp = np.ones_like(img) * multiplier

    return img,exp
    
    
def cncpt_neg(c_expluding, c_id_only):

    tmp = np.zeros((6,6,3))
    tmp_gt_exp = np.zeros((6,6,3))
    tmp_gt_concepts = np.zeros((5,6,6,3)).astype(bool) 

    

    if c_id_only[0] == 1:

        # (¬CP6  v  ¬CP9  v  ¬CP3  v  ¬CP0)
        img_0, _,_ = cncpt(c_expluding, True)
        _, exp_1,_ = cncpt(c_id_only, False)

        if settings_global['stage'] >= 2:
            multiplier_stage = 1/4
        else:
            multiplier_stage = 1

        for y in range(2):
            for x in range(2):
                if abs(exp_1[y*3:(y+1)*3,x*3:(x+1)*3,:]).sum() > 0:
                    tmp[y*3:(y+1)*3,x*3:(x+1)*3,:], tmp_gt_exp[y*3:(y+1)*3,x*3:(x+1)*3,:] = calc_attr_neg_cp(img_0[y*3:(y+1)*3,x*3:(x+1)*3,:], exp_1[y*3:(y+1)*3,x*3:(x+1)*3,:], multiplier_stage , True) 

    elif c_id_only[0] == 3:

        # ¬CP8  v  ¬CP11
        img_0, _,_ =  cncpt(c_expluding, True)
        _, exp_1,_ = cncpt(c_id_only, False)


        if settings_global['stage'] >= 2:
            multiplier_stage = 1/2
        else:
            multiplier_stage = 1

        for y in range(2):
            for x in range(2):
                if abs(exp_1[y*3:(y+1)*3,x*3:(x+1)*3,:]).sum() > 0:

                    tmp[y*3:(y+1)*3,x*3:(x+1)*3,:], tmp_gt_exp[y*3:(y+1)*3,x*3:(x+1)*3,:] = calc_attr_neg_cp(img_0[y*3:(y+1)*3,x*3:(x+1)*3,:], exp_1[y*3:(y+1)*3,x*3:(x+1)*3,:], multiplier_stage , True) 

    else:
        assert False



    return tmp, tmp_gt_exp, tmp_gt_concepts





def cncpt(arr, enabled = True):

    id = random.choice(arr)

    tmp = np.zeros((6,6,3))
    tmp_gt_exp = np.zeros((6,6,3))
    tmp_gt_concepts = np.zeros((5,6,6,3)).astype(bool) 

    if id == 0:
        
        items0 = [random.choice([0,1])]
        items1 = [random.choice([0,1])]
        items = list(set(items0+items1))
        
        if 0 in items:
            tmp[0:3,0:3,0], tmp_gt_exp[0:3,0:3,0] = calc_attr_pos_cp(gen_concept_part_example(7), (1/2) / len(items) , enabled)
            tmp[0:3,3:,0],tmp_gt_exp[0:3,3:,0] = calc_attr_pos_cp(gen_concept_part_example(10), (1/2) / len(items) , enabled)

        if 1 in items:
            tmp[3:,0:3,0],tmp_gt_exp[3:,0:3,0] = calc_attr_pos_cp(gen_concept_part_example(4), (1/2) / len(items) , enabled)
            tmp[3:,3:,0],tmp_gt_exp[3:,3:,0] = calc_attr_pos_cp(gen_concept_part_example(1), (1/2) / len(items) , enabled)
            

    if id == 1:
        tmp[0:3,0:3,0],tmp_gt_exp[0:3,0:3,0] = calc_attr_pos_cp(gen_concept_part_example(6), 1/4 , enabled)
        tmp[0:3,3:,0],tmp_gt_exp[0:3,3:,0] = calc_attr_pos_cp(gen_concept_part_example(9), 1/4 , enabled)
        tmp[3:,0:3,0],tmp_gt_exp[3:,0:3,0] = calc_attr_pos_cp(gen_concept_part_example(3), 1/4 , enabled)
        tmp[3:,3:,0],tmp_gt_exp[3:,3:,0] = calc_attr_pos_cp(gen_concept_part_example(0), 1/4 , enabled)
  
    if id == 2:
        
        items0 = [random.choice([0,1])]
        items1 = [random.choice([0,1])]
        items = list(set(items0+items1))

        if settings_global['stage'] >= 2:
            multiplier_stage = 1/3
        else:
            multiplier_stage = 1


        tmp[3:,0:3,2],tmp_gt_exp[3:,0:3,2] = calc_attr_pos_cp(gen_concept_part_example(4), multiplier_stage , enabled)
        tmp[3:,3:,2],tmp_gt_exp[3:,3:,2] = calc_attr_pos_cp(gen_concept_part_example(1), multiplier_stage , enabled)

        if settings_global['stage'] >= 2:
            multiplier_stage = (1/3) / len(items)
        else:
            multiplier_stage = 1

        if 0 in items:
            tmp[0:3,0:3,1],tmp_gt_exp[0:3,0:3,1] = calc_attr_pos_cp(gen_concept_part_example(7), multiplier_stage , enabled)

        if 1 in items:
            tmp[0:3,3:,1],tmp_gt_exp[0:3,3:,1] = calc_attr_pos_cp(gen_concept_part_example(10), multiplier_stage , enabled)
            
            
    if id == 3: 

        if settings_global['stage'] >= 2:
            multiplier_stage = 1/2
        else:
            multiplier_stage = 1

        tmp[0:3,0:3,1], tmp_gt_exp[0:3,0:3,1] = calc_attr_pos_cp(gen_concept_part_example(8), multiplier_stage , enabled)
        tmp[0:3,3:,1], tmp_gt_exp[0:3,3:,1] = calc_attr_pos_cp(gen_concept_part_example(11), multiplier_stage , enabled)
        
    if id == 4:
        if random.uniform(0, 1) > 0.5:
            img_0, exp_0 = gen_concept_part_example(8) 
            img_1, exp_1 = gen_concept_part_example(11) 
            tmp[0:3,0:3,1], tmp_gt_exp[0:3,0:3,1] = calc_attr_pos_cp([img_0, exp_0], 1/2, enabled)

            _ , tmp_gt_exp[0:3,3:,1] = calc_attr_neg_cp(np.zeros((3,3)) ,exp_1, 1/2, enabled)
        else:
            img_0, exp_0  = gen_concept_part_example(11) 
            img_1, exp_1 = gen_concept_part_example(8) 
            tmp[0:3,3:,1], tmp_gt_exp[0:3,3:,1] = calc_attr_pos_cp([img_0, exp_0], 1/2, enabled)
            _ , tmp_gt_exp[0:3,0:3,1] = calc_attr_neg_cp(np.zeros((3,3)) ,exp_1, 1/2, enabled)

    tmp_gt_concepts[id,:,:,:] = tmp_gt_exp.astype(bool)

    return tmp, tmp_gt_exp, tmp_gt_concepts
    


def gen_class_examples(conceptid, classid, numexamples):
    
    random.seed(0)

    output = []
    gt_exp = []
    gt_concepts = []
    
    c = [0,1,2,3,4] # concepten
    c_id_only = [conceptid]
    c_expluding = [0,1,2,3,4]
    c_expluding.pop(conceptid)
    

        
    for i in range(numexamples):
    
        tmp = np.zeros((6*3,6*3,3))
        tmp_gt_exp = np.zeros((6*3,6*3,3))
        tmp_gt_concepts = np.zeros((5,6*3,6*3,3))
    
        if classid % 5 == 0:
        
            tmp[0:6,0:6,:], tmp_gt_exp[0:6,0:6,:],_ = cncpt(c_id_only)
            tmp[6:12,0:6,:],_,_ = cncpt(c_expluding)
            tmp[12:,0:6,:],_,_ = cncpt(c)

            tmp[0:6,6:12,:],_,_ = cncpt(c_expluding)
            tmp[6:12,6:12,:],_,_ = cncpt(c)
            tmp[12:,6:12,:],_,_ = cncpt(c)

            tmp[0:6,12:,:],_,_ = cncpt(c_expluding)
            tmp[6:12,12:,:],_,_ = cncpt(c)
            tmp[12:,12:,:],_,_ = cncpt(c)

        if classid % 5 == 1:
     
            tmp[0:6,0:6,:],_ ,_ = cncpt(c_expluding)
            tmp[6:12,0:6,:],tmp_gt_exp[6:12,0:6,:],_ = cncpt(c_id_only)
            tmp[12:,0:6,:],_,_= cncpt(c)

            tmp[0:6,6:12,:],_,_ = cncpt(c_expluding)
            tmp[6:12,6:12,:],_,_ = cncpt(c)
            tmp[12:,6:12,:],_,_ = cncpt(c)

            tmp[0:6,12:,:],_,_ = cncpt(c_expluding)
            tmp[6:12,12:,:],_,_ = cncpt(c)
            tmp[12:,12:,:],_,_ = cncpt(c)
            
                
        if classid % 5 == 2:

            if settings_global['stage'] >= 1:
                multiplier_stage = 1/2
            else:
                multiplier_stage = 1


            tmp[0:6,0:6,:],_,_ = cncpt(c_expluding)
            tmp[6:12,0:6,:],_,_ = cncpt(c_expluding)
            tmp[12:,0:6,:],_,_ = cncpt(c)

            tmp[0:6,6:12,:],tmp_gt_exp[0:6,6:12,:],_ = cncpt(c_id_only)
            tmp_gt_exp[0:6,6:12,:] = tmp_gt_exp[0:6,6:12,:] * multiplier_stage
            tmp[6:12,6:12,:],_,_ = cncpt(c)
            tmp[12:,6:12,:],_ ,_ = cncpt(c)

            tmp[0:6,12:,:],tmp_gt_exp[0:6,12:,:],_ = cncpt(c_id_only)
            tmp_gt_exp[0:6,12:,:] = tmp_gt_exp[0:6,12:,:] * multiplier_stage
            tmp[6:12,12:,:],_,_ = cncpt(c)
            tmp[12:,12:,:],_,_= cncpt(c)
       
        if classid % 5 == 3:

            if settings_global['stage'] >= 1:
                multiplier_stage = 1/2
            else:
                multiplier_stage = 1


            if random.uniform(0, 1) > 0.5:
                   
                tmp[0:6,0:6,:],_,_ = cncpt(c_expluding)
                tmp[6:12,0:6,:],_,_ = cncpt(c_expluding)
                tmp[12:,0:6,:],_,_ = cncpt(c)

    
                tmp[0:6,6:12,:], tmp_gt_exp[0:6,6:12,:],_ = cncpt_neg(c_expluding, c_id_only)
                tmp_gt_exp[0:6,6:12,:] = tmp_gt_exp[0:6,6:12,:] * multiplier_stage
                tmp[6:12,6:12,:],_ ,_ = cncpt(c)
                tmp[12:,6:12,:],_,_ = cncpt(c)


                tmp[0:6,12:,:],tmp_gt_exp[0:6,12:,:],_ = cncpt(c_id_only)

                tmp_gt_exp[0:6,12:,:] = tmp_gt_exp[0:6,12:,:] * multiplier_stage
                tmp[6:12,12:,:],_,_ = cncpt(c)
                tmp[12:,12:,:],_ ,_ =cncpt(c)

            else:

                tmp[0:6,0:6,:],_,_ = cncpt(c_expluding)
                tmp[6:12,0:6,:],_,_ = cncpt(c_expluding)
                tmp[12:,0:6,:],_ ,_ = cncpt(c)


                tmp[0:6,6:12,:],tmp_gt_exp[0:6,6:12,:],_ = cncpt(c_id_only)
                tmp_gt_exp[0:6,6:12,:] = tmp_gt_exp[0:6,6:12,:] * multiplier_stage
                tmp[6:12,6:12,:],_,_ = cncpt(c)
                tmp[12:,6:12,:],_,_ = cncpt(c)
     

                tmp[0:6,12:,:] , tmp_gt_exp[0:6,12:,:],_ = cncpt_neg(c_expluding, c_id_only)
                tmp_gt_exp[0:6,12:,:] = tmp_gt_exp[0:6,12:,:] * multiplier_stage
                 
                tmp[6:12,12:,:],_,_= cncpt(c)
                tmp[12:,12:,:],_ ,_ = cncpt(c)

        if classid % 5 == 4:
   
            tmp[0:6,0:6,:] , tmp_gt_exp[0:6,0:6,:],_ = cncpt_neg(c_expluding, c_id_only)
            tmp_gt_exp[0:6,0:6,:] = tmp_gt_exp[0:6,0:6,:] * 1/4
            tmp[6:12,0:6,:] , tmp_gt_exp[6:12,0:6,:],_ = cncpt_neg(c_expluding, c_id_only)
            tmp_gt_exp[6:12,0:6,:] = tmp_gt_exp[6:12,0:6,:] * 1/4
            tmp[12:,0:6,:],_ ,_ = cncpt(c)
            


            tmp[0:6,6:12,:] , tmp_gt_exp[0:6,6:12,:],_ = cncpt_neg(c_expluding, c_id_only)
            tmp_gt_exp[0:6,6:12,:] = tmp_gt_exp[0:6,6:12,:] * 1/4
            tmp[6:12,6:12,:],_,_ = cncpt(c)
            tmp[12:,6:12,:],_,_ = cncpt(c)
  


            tmp[0:6,12:,:] , tmp_gt_exp[0:6,12:,:],_ = cncpt_neg(c_expluding, c_id_only)
            tmp_gt_exp[0:6,12:,:] = tmp_gt_exp[0:6,12:,:] * 1/4
            tmp[6:12,12:,:],_,_ = cncpt(c)
            tmp[12:,12:,:],_,_ = cncpt(c)

        output.append(tmp)
        
        gt_exp.append(tmp_gt_exp)

        gt_concepts.append(tmp_gt_concepts.astype(bool))
            



    gt_exp = np.array(gt_exp)

    if settings_global['stage'] >= 4:
        # normalize explanations
        for i in range(len(gt_exp)):
            gt_exp[i] = normalize_explanation(gt_exp[i])



    return np.array(output), gt_exp, np.array(gt_concepts)


def convert_to_2D(explanation): 
    output = np.zeros((explanation.shape[0], explanation.shape[1]))
    for y in range(explanation.shape[0]):
        for x in range(explanation.shape[1]):
            mostExtremeValue = 0.0

            for c in range(3):
                if abs(explanation[y,x,c]) > abs(mostExtremeValue):
                    if mostExtremeValue != 0.0:
                        print("ERROR!!!!! mostExtremeValue != 0")
                        exit()
                    mostExtremeValue = explanation[y,x,c]
                    
            output[y,x] = mostExtremeValue
            
    return output


def upscale(img):
    new_img = np.zeros((18*4,18*4,3))

    for y in range(18):
        for x in range(18):              
            for r_0 in range(4):
                for r_1 in range(4):   
                    new_img[y*4+r_0,x*4+r_1,:] = img[y,x,:]

    return new_img

def cutout(img):
    return img[:24,24:,:]



for settings_global['stage'] in range(1,5,1):

    # example left
    inputs, exps, _ = gen_class_examples(3, 3, 1)

    exps[exps>1] = 1
    exps[exps<-1] = -1
    
    saveImage("right_concept_3_class_3_stage_"+str(settings_global['stage'])+"_input.png", cutout(upscale(inputs[0])))
    saveImageColomap("right_concept_3_class_3_stage_"+str(settings_global['stage'])+"_exp.png", cutout(upscale(exps[0])))

    # example right
    inputs, exps, _ = gen_class_examples(2, 2, 1)

    exps[exps>1] = 1
    exps[exps<-1] = -1
    
    saveImage("left_concept_2_class_2_stage_"+str(settings_global['stage'])+"_input.png", cutout(upscale(inputs[0])))
    saveImageColomap("left_concept_2_class_2_stage_"+str(settings_global['stage'])+"_exp.png", cutout(upscale(exps[0])))





