
# ---------------------- Import Libs ---------------------- #

from dataset_synth import dataset_synth

import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
import re






def conv_copy(name,value,model):

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0
    
    for chan in range(np.min([weigths[0].shape[2],weigths[0].shape[3]])):
        if weigths[0].shape[0] == 3:
            weigths[0][1,1,chan,chan] = value 
        else:
            weigths[0][:,:,chan,chan] = value 
        weigths[1][chan] = 0

    model.get_layer(name).set_weights(weigths)

def conv_filter(name,value,model):

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0
    
    for chan in range(48):
        if weigths[0].shape[0] == 3:
            weigths[0][1,1,chan,chan] = value 
        else:
            weigths[0][:,:,chan,chan] = value 
        weigths[1][chan] = 0

    model.get_layer(name).set_weights(weigths)



def conv_move_toback(name,value,model,multiplyer):
    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0
        
    for chan in range(64):
        if weigths[0].shape[0] == 3:
            weigths[0][1,1,chan,chan+64*multiplyer] = value 
        else:
            weigths[0][:,:,chan,chan+64*multiplyer] = value 
        weigths[1][chan] = 0

    model.get_layer(name).set_weights(weigths)
    
    
def conv_move_fromback(name,value,model,multiplyer):
    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for chan in range(64):
        if weigths[0].shape[0] == 3:
            weigths[0][1,1,chan+64*multiplyer,chan] = value 
        else:
            weigths[0][:,:,chan+64*multiplyer,chan] = value 
        weigths[1][chan] = 0

    model.get_layer(name).set_weights(weigths)
    
    

  
def convThreeOnes(name,value,model,offset=0):

    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for chan in range(48):
        if weigths[0].shape[0] != 3:
            print("ERROR!")
            exit()
        weigths[0][:,:,chan,chan+offset] = value 
        weigths[0][1,1,chan,chan+offset] = 0
        weigths[1][chan] = -3

    model.get_layer(name).set_weights(weigths)

def convThreeOnesMove(name,value,model):

    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for chan in range(weigths[0].shape[2]):
        weigths[0][2,2,chan,chan] = value 
        weigths[1][chan] = 0

    model.get_layer(name).set_weights(weigths)
    

    
def batchNormOnes(name,model):
    weigths = model.get_layer(name).get_weights()
    weigths[0][:] = 1 
    weigths[1][:] = 0
    weigths[2][:] = 0
    weigths[3][:] = 1
    model.get_layer(name).set_weights(weigths)




# source: https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model 
def insert_layer_nonseq(model, insert_layer_name=None, position='after',lastLayerNum=-1):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:lastLayerNum]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match('conv1_bn', layer.name):
 

            # ****************************************************************************************
            # *                Mask to emulate strides (conv1_bn -> lambda)
            # ****************************************************************************************
         
            x = layer_input

            new_layer = keras.layers.Lambda(lambda x: x[0]*x[1])
           
           
            mask = np.zeros((112, 112, 64))
            the_range = [((3+a*4)+b*16-1) for a in range(3) for b in range(7)]
            for y2 in range(112):
                for x2 in range(112):
                    if (y2 in the_range) and (x2 in the_range):
                        mask[y2,x2,:] = 1
           
            x = new_layer([x,mask])
            
        elif re.match('conv2_block3_3_bn', layer.name):
 
            # ****************************************************************************************
            # *                Mask to emulate strides (conv2_block3_3_bn -> lambda_1)
            # ****************************************************************************************
 
            x = layer_input

            new_layer = keras.layers.Lambda(lambda x: x[0]*x[1])
           
           
            mask = np.zeros((56, 56, 256))
            the_x_range = [( 3+x*8 ) for x in range(7) ]
            the_y_range = [( 3+x*8 ) for x in range(7) ]
            for y2 in range(56):
                for x2 in range(56):
                    if (y2 in the_y_range) and (x2 in the_x_range):
                        mask[y2,x2,:] = 1
           
            x = new_layer([x,mask])
            
        elif re.match('conv3_block1_1_conv', layer.name): 
        
            # ****************************************************************************************
            # *              Kernel size of (2,2) instead of (1,1)   (conv3_block1_1_conv)
            # ****************************************************************************************
            
            new_layer = keras.layers.Conv2D(128,(2,2),strides=(2,2),padding="valid",name="conv3_block1_1_conv")
      
            x = layer_input
            
            x = new_layer(x)
            
        else:
        
            # ****************************************************************************************
            # *                ReLU layers have custom maximum value of 1
            # ****************************************************************************************
            
            if ("_relu" in layer.name) or ("_out" in layer.name):
                layer = keras.layers.ReLU(max_value=1,name=layer.name) 
        
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        
    model_outputs.append(x)

    return keras.Model(inputs=model.inputs, outputs=model_outputs)

def compile_model(model):
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['acc'])    


def genModel(conceptid, block_id=17,layer_id=-1):
    print("# ---------------------- Define Model ---------------------- #")

    #### Concept classification


    block_layers = [6,18, 28, 38,50,60,70,80,92,102,112,122,132,142,154,164,174,-2]
    
    print("block_id",block_id)

    model = applications.ResNet50(weights=None)

    lastLayerNum = block_layers[block_id]+1
    
    if block_id==2 and layer_id != -1:
        lastLayerNum = [21,24][layer_id]+1

    if block_id==0 and layer_id != -1:
        lastLayerNum = 3

    model = insert_layer_nonseq(model,  insert_layer_name=None, position='replace', lastLayerNum=lastLayerNum) 
   


    # # ---------------------- Set weights of model ---------------------- #
    print("# ---------------------- Set weights of model ---------------------- #")
    
    
    used_indices = [7, 22, 28, 37, 6, 21, 27, 36, 19, 34, 52, 61, 20, 35,  46,76,109,45,75,108,58,100,133,59]
 
    
    def getcpchannel(pc,pos,channel):
        index = pc + channel*numPc + pos*numchannels*numPc
        
        if index in used_indices:
            return used_indices.index(index)
        else:
            print("Missing",index)

    # ****************************************************************************************
    # *                                 Concept Detectors (conv1_conv)
    # ****************************************************************************************

    weigths = model.get_layer('conv1_conv').get_weights()
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    posLookup = [[[0,3],[0,3]],[[0,3],[4,7]],[[4,7],[0,3]],[[4,7],[4,7]]]
    numPc = 12 
    numchannels = 3
   

    for channel in range(numchannels):
        for pc in range(numPc): 
            for pos in range(4):
                index = pc + channel*numPc + pos*numchannels*numPc
                if index in used_indices:
                    weigths[0][posLookup[pos][0][0]:posLookup[pos][0][1], posLookup[pos][1][0]:posLookup[pos][1][1], channel, getcpchannel(pc,pos,channel)] = dataset_synth.gen_concept_part_weights(pc)
                    weigths[1][getcpchannel(pc,pos,channel)] = -1.0
    
    model.get_layer('conv1_conv').set_weights(weigths)

    if block_id == 0 and layer_id == 0:
        compile_model(model)
        return model

    if block_id == 0:
        compile_model(model)
        return model


    batchNormOnes('conv2_block1_0_bn',model)


    # ****************************************************************************************
    # *          concept definition a,b without 'not' in c,d (conv2_block1_0_conv)
    # ****************************************************************************************
    
    name = 'conv2_block1_0_conv'

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    # 0
    weigths[0][0,0,getcpchannel(7,0,0),0] = 1
    weigths[0][0,0,getcpchannel(10,1,0),0] = 1
    weigths[1][0] = -1
    
    #1
    weigths[0][0,0,getcpchannel(4,2,0),1] = 1
    weigths[0][0,0,getcpchannel(1,3,0),1] = 1
    weigths[1][1] = -1

    #2
    weigths[0][0,0,getcpchannel(6,0,0),2] = 1
    weigths[0][0,0,getcpchannel(9,1,0),2] = 1
    weigths[1][2] = -1

    #3
    weigths[0][0,0,getcpchannel(3,2,0),3] = 1
    weigths[0][0,0,getcpchannel(0,3,0),3] = 1
    weigths[1][3] = -1
    
    #4
    weigths[0][0,0,getcpchannel(7,0,1),4] = 1
    weigths[0][0,0,getcpchannel(10,1,1),4] = 1
    weigths[1][4] = 0
    
    #5
    weigths[0][0,0,getcpchannel(4,2,2),5] = 1
    weigths[0][0,0,getcpchannel(1,3,2),5] = 1
    weigths[1][5] = -1
        
    #7
    weigths[0][0,0,getcpchannel(8,0,1),7] = 1
    weigths[0][0,0,getcpchannel(11,1,1),7] = 1
    weigths[1][7] = 0

    model.get_layer(name).set_weights(weigths)



    batchNormOnes('conv2_block1_1_bn',model)


    # ****************************************************************************************
    # *          concept definition a,b with 'not' in c,d (conv2_block1_1_conv)
    # ****************************************************************************************
    
    
    name = 'conv2_block1_1_conv'

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0
    
    #6
    weigths[0][0,0,getcpchannel(8,0,1),6] = 1
    weigths[0][0,0,getcpchannel(11,1,1),6] = 1
    weigths[1][6] = -1
   
    model.get_layer(name).set_weights(weigths)

   

    # ****************************************************************************************
    # *          concept definition c,d (conv2_block1_2_conv)
    # ****************************************************************************************

    batchNormOnes('conv2_block1_2_bn',model)


    name = 'conv2_block1_2_conv'

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    #6 copy
    weigths[0][1,1,6,6] = 1
    weigths[1][6] = 0

    #6 not 6
    weigths[0][1,1,6,8] = -1
    weigths[1][8] = 1

    model.get_layer(name).set_weights(weigths)
    
    batchNormOnes('conv2_block1_3_bn',model)
    conv_copy('conv2_block1_3_conv', 1,model)

    if block_id == 1:
        compile_model(model)
        return model



    batchNormOnes('conv2_block2_1_bn',model)



    # ****************************************************************************************
    # *                   concept definition e (conv2_block2_1_conv)
    # ****************************************************************************************

    name = 'conv2_block2_1_conv'

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    # 0: 0 OR 1
    weigths[0][0,0,0,9] = 1
    weigths[0][0,0,1,9] = 1
    weigths[1][9] = 0
    
    # 1: 2 AND 3
    weigths[0][0,0,2,10] = 1
    weigths[0][0,0,3,10] = 1
    weigths[1][10] = -1

    # 2: 4 AND 5
    weigths[0][0,0,4,11] = 1
    weigths[0][0,0,5,11] = 1
    weigths[1][11] = -1

    # 3: 6
    weigths[0][0,0,6,12] = 1
    weigths[1][12] = 0
    
    # 4: 7 AND 8
    weigths[0][0,0,7,13] = 1
    weigths[0][0,0,8,13] = 1
    weigths[1][13] = -1

    model.get_layer(name).set_weights(weigths)
    
    
    if block_id == 2 and layer_id == 0:
        compile_model(model)
        return model
   
    
    batchNormOnes('conv2_block2_2_bn',model)

    
    # ****************************************************************************************
    # *                   class definition a,b (conv2_block2_2_conv)
    # ****************************************************************************************

    name = 'conv2_block2_2_conv'

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    # 0: 0
    weigths[0][0,0,conceptid+9,14] = 1
    weigths[1][14] = 0

    # 1: 3
    weigths[0][2,0,conceptid+9,15] = 1
    weigths[1][15] = 0

    # 2: 1 AND 2
    weigths[0][0,0,conceptid+9,16] = 1
    weigths[0][0,2,conceptid+9,16] = 1
    weigths[1][16] = -1
    
    # 3: 1 OR 2
    weigths[0][0,0,conceptid+9,17] = 1
    weigths[0][0,2,conceptid+9,17] = 1
    weigths[1][17] = 0

    # 4: 0 OR 3
    weigths[0][0,0,conceptid+9,18] = 1
    weigths[0][2,0,conceptid+9,18] = 1
    weigths[1][18] = 0
    
    # 5: 1 OR 2
    weigths[0][0,0,conceptid+9,19] = 1
    weigths[0][0,2,conceptid+9,19] = 1
    weigths[1][19] = 0
    

    model.get_layer(name).set_weights(weigths)
    
    
    if block_id == 2 and layer_id == 1:
        compile_model(model)
        return model
    
    batchNormOnes('conv2_block2_3_bn',model)
    
    # ****************************************************************************************
    # *                   class definition c,d (conv2_block2_3_conv)
    # ****************************************************************************************

    name = 'conv2_block2_3_conv'

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0
    
    for channel in range(14,20):
        weigths[0][0,0,channel,channel] = 1

    # not 1 and 2
    weigths[0][0,0,16,20] = -1
    weigths[1][20] = 1

    # not 0 OR 3
    weigths[0][0,0,18,21] = -1
    weigths[1][21] = 1
    
    # not 1 OR 2
    weigths[0][0,0,19,22] = -1
    weigths[1][22] = 1
    
    model.get_layer(name).set_weights(weigths)


        
    
    if block_id == 2:
        compile_model(model)
        return model



    batchNormOnes('conv2_block3_1_bn',model)
    batchNormOnes('conv2_block3_2_bn',model)
    #batchNormOnes('conv2_block3_3_bn',model)

    conv_copy('conv2_block3_1_conv',1,model)
    
    
    # ****************************************************************************************
    # *                   class definition e (conv2_block3_2_conv)
    # ****************************************************************************************
    
    name = 'conv2_block3_2_conv'

    weigths = model.get_layer(name).get_weights()

    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0
    

    # 0: 0
    weigths[0][0,0,14,23] = 1
    weigths[1][23] = 0

    # 1: 3
    weigths[0][0,0,15,24] = 1
    weigths[1][24] = 0

    # 2: 1 AND 2
    weigths[0][0,2,16,25] = 1
    weigths[1][25] = 0
    
    # 3: 1 OR 2 AND not 1 AND 2
    weigths[0][0,2,17,26] = 1
    weigths[0][0,2,20,26] = 1
    weigths[1][26] = -1

    # 4: not 0 OR 3 and not 1 OR 2
    weigths[0][0,0,21,27] = 1
    weigths[0][0,2,22,27] = 1
    weigths[1][27] = -1
    
    model.get_layer(name).set_weights(weigths)
        
    
    

    name = 'conv2_block3_3_conv'
    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0


    for chan in range(5):
        weigths[0][0,0,chan+23,chan+23] = 1 
    model.get_layer(name).set_weights(weigths)
    
    

    if block_id == 3:
        compile_model(model)
        return model




    prefixes = [["conv3_block1","conv3_block2","conv3_block3"],["conv3_block4"],["conv4_block1","conv4_block2","conv4_block3"],["conv4_block4","conv4_block5","conv4_block6"]]
    block_nums = [[4,5,6],[7],[8,9,10],[11,12,13]]
        
    for idx in range(len(prefixes)):
        if idx == 1:
            batchNormOnes('conv3_block4_1_bn',model)
            batchNormOnes('conv3_block4_2_bn',model)
            batchNormOnes('conv3_block4_3_bn',model)

            conv_copy('conv3_block4_1_conv',1,model)
            convThreeOnesMove('conv3_block4_2_conv',1,model)
            conv_copy('conv3_block4_3_conv',1,model)
            
            if block_id == block_nums[idx][0]:
                compile_model(model)
                return model
            
        else:
            if idx == 0 or idx == 2:
                batchNormOnes(prefixes[idx][0]+'_0_bn',model)
                conv_copy(prefixes[idx][0]+'_0_conv',0,model)
                
            batchNormOnes(prefixes[idx][0]+'_1_bn',model)
            conv_copy(prefixes[idx][0]+'_1_conv',1,model)
            
            if idx == 0:
                batchNormOnes(prefixes[idx][0]+'_2_bn',model)
                
                name = prefixes[idx][0]+'_2_conv'
                weigths = model.get_layer(name).get_weights()
              
                weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
                weigths[1][:] = 0

                for chan in range(weigths[0].shape[3]):
                    weigths[0][1,1,chan,chan] = 1 
                    weigths[1][chan] = 0

                model.get_layer(name).set_weights(weigths)
            else:
                batchNormOnes(prefixes[idx][0]+'_2_bn',model)
                conv_copy(prefixes[idx][0]+'_2_conv',1,model)
            
            batchNormOnes(prefixes[idx][0]+'_3_bn',model)
            conv_copy(prefixes[idx][0]+'_3_conv',1,model)
            
            if block_id == block_nums[idx][0]:
                compile_model(model)
                return model
                
                

            # ****************************************************************************************
            # *                           (c_(n+1) OR not c_(n+1)) AND c_(n)
            # ****************************************************************************************
                    
   
            batchNormOnes(prefixes[idx][1]+'_1_bn',model)

            name = prefixes[idx][1]+'_1_conv'
            weigths = model.get_layer(name).get_weights()
          
            weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
            weigths[1][:] = 0

            # c
            for chan in range(5):
                weigths[0][0,0,23+chan,23+chan] = 1
                    
            # not c
            for chan in range(5):
                weigths[0][0,0,23+chan,23+chan+5] = -1
                weigths[1][23+chan+5] = 1
                
            model.get_layer(name).set_weights(weigths)
            
            
            
            
        
            batchNormOnes(prefixes[idx][1]+'_2_bn',model)

            name = prefixes[idx][1]+'_2_conv'
            weigths = model.get_layer(name).get_weights()
          
            weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
            weigths[1][:] = 0

            for chan in range(5):
            
                weigths[0][1,1,23+chan,23+chan] = 1
            
                weigths[0][1,1,23+chan,64+chan] = 1
                weigths[0][1,1,23+chan+5,64+chan] = 1
                weigths[1][64+chan] = 0
                
            model.get_layer(name).set_weights(weigths)
            
            
            
            batchNormOnes(prefixes[idx][1]+'_3_bn',model)

            name = prefixes[idx][1]+'_3_conv'
            weigths = model.get_layer(name).get_weights()
          
            weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
            weigths[1][:] = 0

            for chan in range(4):
                weigths[0][0,0,23+chan,23+chan] = 1
                weigths[0][0,0,64+chan+1,23+chan] = 1
                weigths[1][23+chan] = -1
            weigths[0][0,0,23+4,23+4] = 1
            weigths[0][0,0,64+0,23+4] = 1
            weigths[1][23+4] = -1
            
                
            model.get_layer(name).set_weights(weigths)
            
            if block_id == block_nums[idx][1]:
                compile_model(model)
                return model
            
            
            
            # ****************************************************************************************
            # *                              (C_n OR c_(n+1)) AND not (c_(n+1))
            # ****************************************************************************************
                    
            batchNormOnes(prefixes[idx][2]+'_1_bn',model)

            name = prefixes[idx][2]+'_1_conv'
            weigths = model.get_layer(name).get_weights()
          
            weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
            weigths[1][:] = 0

            # c
            for chan in range(5):
                weigths[0][0,0,23+chan,23+chan] = 1
                    
            # not c
            for chan in range(5):
                weigths[0][0,0,23+chan,23+chan+5] = -1
                weigths[1][23+chan+5] = 1
                
            model.get_layer(name).set_weights(weigths)
            
            
            
            
        
            batchNormOnes(prefixes[idx][2]+'_2_bn',model)

            name = prefixes[idx][2]+'_2_conv'
            weigths = model.get_layer(name).get_weights()
          
            weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
            weigths[1][:] = 0

            for chan in range(5):
                weigths[0][1,1,23+chan,23+chan] = 1
                
            for chan in range(4):
                weigths[0][1,1,23+chan+5+1,23+chan] = 1
            chan = 4
            weigths[0][1,1,23+5,23+chan] = 1
                
            model.get_layer(name).set_weights(weigths)
            
            
            
            
            
            
            batchNormOnes(prefixes[idx][2]+'_3_bn',model)

            name = prefixes[idx][2]+'_3_conv'
            weigths = model.get_layer(name).get_weights()
          
            weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
            weigths[1][:] = 0

            for chan in range(5):
                weigths[0][0,0,23+chan,23+chan] = 1
                weigths[0][0,0,23+chan+5+1,23+chan] = 1
                weigths[1][23+chan] = -1
                
            model.get_layer(name).set_weights(weigths)
            
            
            
            
            
            if block_id == block_nums[idx][2]:
                compile_model(model)
                return model
            
            
            
            
            
 
 
 
 
 
 
 
 
    batchNormOnes('conv5_block1_0_bn',model)
    batchNormOnes('conv5_block1_1_bn',model)
    batchNormOnes('conv5_block1_2_bn',model)
    batchNormOnes('conv5_block1_3_bn',model)



    # ****************************************************************************************
    # *   move classes to channel 64 + 0-5, set others to 0 (conv5_block1_0_conv, conv5_block1_1_conv)
    # ****************************************************************************************


    conv_copy('conv5_block1_0_conv',0,model)


    name = 'conv5_block1_1_conv'
    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = -1000

    for chan in range(5):
        weigths[0][0,0,23+chan,chan+64] = 1 
        weigths[1][chan+64] = 0

    model.get_layer(name).set_weights(weigths)
    

    # ****************************************************************************************
    # *                Ignore bottom and right row/column (conv5_block1_2_conv)
    # ****************************************************************************************

    name = 'conv5_block1_2_conv'
    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for chan in range(5):
        weigths[0][0,0,chan+64,chan+32] = 1 
        weigths[1][chan+32] = 0

    model.get_layer(name).set_weights(weigths)
    
    
    
    
    
    conv_copy('conv5_block1_3_conv',1,model)

    if block_id == 14:
        compile_model(model)
        return model


    batchNormOnes('conv5_block2_1_bn',model)
    batchNormOnes('conv5_block2_2_bn',model)
    batchNormOnes('conv5_block2_3_bn',model)

    conv_copy('conv5_block2_1_conv',1,model)



    # ****************************************************************************************
    # *                AND (conv5_block2_2_conv)
    # ****************************************************************************************

    name = 'conv5_block2_2_conv'
    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for chan in range(5):
        weigths[0][:,:,chan+32,chan+64] = 1 
        weigths[0][:,:,chan+32,chan+32] = 1 
        weigths[1][chan+64] = -8

    model.get_layer(name).set_weights(weigths)
    
    
    
    
    # ****************************************************************************************
    # *                set channel + 32 to 0 (conv5_block2_3_conv)
    # ****************************************************************************************
    
    name = 'conv5_block2_3_conv'
    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for chan in range(5):
        weigths[0][0,0,chan+64,chan+64] = 1 
        weigths[0][0,0,chan+32,chan+32] = -1 

    model.get_layer(name).set_weights(weigths)
    
    
    
    

    if block_id == 15:
        compile_model(model)
        return model
        

    batchNormOnes('conv5_block3_1_bn',model)
    batchNormOnes('conv5_block3_2_bn',model)
    batchNormOnes('conv5_block3_3_bn',model)

    conv_copy('conv5_block3_1_conv',1,model)


    
    # ****************************************************************************************
    # *  AND:  move to chan + 0 (conv5_block3_2_conv)
    # ****************************************************************************************
    
    name = 'conv5_block3_2_conv'
    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for chan in range(5):
        weigths[0][1,1,chan+64,chan+64] = 1 
        
        weigths[0][:,:,chan+64,chan] = 1
        weigths[1][chan] = -8
        

    model.get_layer(name).set_weights(weigths)
    
    
    

    # ****************************************************************************************
    # *                set channel + 64 to 0 (conv5_block3_3_conv)
    # ****************************************************************************************

    name = 'conv5_block3_3_conv'
    weigths = model.get_layer(name).get_weights()
  
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for chan in range(5):
        weigths[0][0,0,chan,chan] = 1 
        weigths[0][0,0,chan+64,chan+64] = -1 

    model.get_layer(name).set_weights(weigths)
    
    


    if block_id == 16:
        compile_model(model)
        return model

    
    if False:
        #tf.keras.utils.plot_model(model)
        if True:
            layer_array = []

            for layer in model.layers:
                #print(layer.__class__.__name__)
                if layer.__class__.__name__ == "Conv2D":
                    layer_array.append([layer.name,layer.__class__.__name__ , str(layer.kernel_size), str(layer.strides), str(layer.filters), str(layer.output_shape)])
                else:
                    layer_array.append([layer.name,layer.__class__.__name__ ,"","","",str(layer.output_shape)])

            #sorter = lambda x: (x[0].split("_")[0], x[0].split("_")[1], x[0].split("_")[2 if len(x[0].split("_")) > 2 else 1])
            #layer_array = sorted(layer_array, key=sorter)

            print("&".join(["Name","Kernel size","Strides","Filters","Output shape"]),"\\\\")
            prevblock = "-1"
            for line in layer_array:
                if len(line[0].split("_")) > 1:
                    if (not (prevblock == line[0].split("_")[1])):
                        print("\\\\")
                    prevblock = line[0].split("_")[1]
                    
                outstring = "&".join(line)+"\\\\"
                outstring = outstring.replace("_","\_")
                print(outstring)
            
            exit()

    compile_model(model)
    return model
