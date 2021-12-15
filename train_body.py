import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from math import cos, sin
from torch.optim.lr_scheduler import StepLR
from models_body import *

def string_to_list(string, dim):
    string=string[1:-1]
    string=list(map(float,string.split(",")))
    # print(string)
    array=np.array(string)
    array=np.reshape(array,(-1, dim))
    return array


class PoseDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_name, data_type='None', transform=None):
            """
            :param data_folder: folder where data files are stored
            :param data_name: base name of processed datasets
            :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
            :param transform: image transform pipeline
            """

            self.num_input_joints = 49
            self.input_dof = 2 
            self.num_dof = 4
            self.num_output_joints = 14
            self.num_frames = 32

            data=pd.read_csv(data_name)

            # convert the columns from string to list using the "string_to_list" function defined above.
            data['2d_pose']=data.apply(lambda row: string_to_list(row['2d_pose'], 2), axis=1)
            data['rotation_quaternion']=data.apply(lambda row: string_to_list(row['rotation_quaternion'], 4), axis=1)

            # Drop the indices where 2d_pose values does not have a length of 67.
            index_names = data[ data.apply(lambda row: len(row['2d_pose']) != 67, axis=1) ].index
            data.drop(index_names,inplace=True)
            data=data.reset_index(drop=True)
            print("2D pose dropped indices", len(index_names))

            # Drop the indices where quaternion values does not have a length of 52.
            index_names = data[ data.apply(lambda row: len(row['rotation_quaternion']) != 60, axis=1) ].index
            data.drop(index_names,inplace=True)
            data=data.reset_index(drop=True)
            print("Quaternion dropped indices", len(index_names))

            # Delete entries where the 2d_pose values are all 0.
            index_names = data[ data.apply(lambda row: np.sum(row['2d_pose'])==0, axis=1) ].index
            data.drop(index_names,inplace=True)
            data=data.reset_index(drop=True)
            print("Zero entries dropped dropped indices", len(index_names))

            # Drop entries where the 2d_pose values are negetive.
            index_names = data[ data.apply(lambda row: np.sum(row['2d_pose']<0)!=0, axis=1) ].index
            data.drop(index_names,inplace=True)
            data=data.reset_index(drop=True)
            print("Negative entries dropped dropped indices", len(index_names))


            # Max term for normalizing the 2d_pose (1080 and 1920 correspond to frame width and height respectively)
            self.norm_2d=np.array([1080,1920])
            # Drop entries where the 2d_pose values are greater than 1
            index_names = data[ data.apply(lambda row: np.sum(row['2d_pose']/self.norm_2d>1)!=0, axis=1) ].index
            data.drop(index_names,inplace=True)
            data=data.reset_index(drop=True)
            print("Greater than 1 entries dropped dropped indices", len(index_names))

            print("Total number of data: ", len(data))

            # Defining class variables
            self.data_type=data_type
            self.data=data
            self.dataset_size = len(self.data)
            self.transform=transform

            # Hand joints from the shoulder to the fingers
            self.hands=['shoulder.L', 'upperarm.L', 'forearm.L', 'hand.L', 'palm.04.L', 'little.01.L', 'little.02.L', 'little03.L', 'palm.03.L', 'ring.01.L', 'ring.02.L', 'ring.03.L', 'palm.02.L', 'middle.01.L', 'middle.02.L', 'middle.03.L', 'palm.01.L', 'index.01.L', 'index.02.L', 'index.03.L', 'thumb.01.L', 'thumb.02.L', 'thumb.03.L', 'shoulder.R', 'upperarm.R', 'forearm.R', 'hand.R', 'palm.04.R', 'little.01.R', 'little.02.R', 'little03.R', 'palm.03.R', 'ring.01.R', 'ring.02.R', 'ring.03.R', 'palm.02.R', 'middle.01.R', 'middle.02.R', 'middle.03.R', 'palm.01.R', 'index.01.R', 'index.02.R', 'index.03.R', 'thumb.01.R', 'thumb.02.R', 'thumb.03.R']
            
            # Hand joints from the palm to the fingers
            self.palms=[ 'palm.04.L', 'little.01.L', 'little.02.L', 'little03.L', 'palm.03.L', 'ring.01.L', 'ring.02.L', 'ring.03.L', 'palm.02.L', 'middle.01.L', 'middle.02.L', 'middle.03.L', 'palm.01.L', 'index.01.L', 'index.02.L', 'index.03.L', 'thumb.01.L', 'thumb.02.L', 'thumb.03.L', 'palm.04.R', 'little.01.R', 'little.02.R', 'little03.R', 'palm.03.R', 'ring.01.R', 'ring.02.R', 'ring.03.R', 'palm.02.R', 'middle.01.R', 'middle.02.R', 'middle.03.R', 'palm.01.R', 'index.01.R', 'index.02.R', 'index.03.R', 'thumb.01.R', 'thumb.02.R', 'thumb.03.R']

            # Body joints till the wrist
            self.body=['hips', 'spine', 'chest', 'upper-chest','shoulder.L', 'upperarm.L', 'forearm.L', 'hand.L',
                        'neck', 'head','shoulder.R', 'upperarm.R', 'forearm.R', 'hand.R']

            # Complete joint list
            self.joints=['hips', 'spine', 'chest', 'upper-chest', 'shoulder.L', 'upperarm.L', 'forearm.L', 'hand.L', 'palm.04.L', 'little.01.L', 'little.02.L', 'little03.L', 'palm.03.L', 'ring.01.L', 'ring.02.L', 'ring.03.L', 'palm.02.L', 'middle.01.L', 'middle.02.L', 'middle.03.L', 'palm.01.L', 'index.01.L', 'index.02.L', 'index.03.L', 'thumb.01.L', 'thumb.02.L', 'thumb.03.L', 'neck', 'head', 'shoulder.R', 'upperarm.R', 'forearm.R', 'hand.R', 'palm.04.R', 'little.01.R', 'little.02.R', 'little03.R', 'palm.03.R', 'ring.01.R', 'ring.02.R', 'ring.03.R', 'palm.02.R', 'middle.01.R', 'middle.02.R', 'middle.03.R', 'palm.01.R', 'index.01.R', 'index.02.R', 'index.03.R', 'thumb.01.R', 'thumb.02.R', 'thumb.03.R', 'thigh.L', 'shin.L', 'foot.L', 'toe.L', 'thigh.R', 'shin.R', 'foot.R', 'toe.R']
            
            # Upper body joints including fingers
            self.body_hand_joints=['hips', 'spine', 'chest', 'upper-chest', 'shoulder.L', 'upperarm.L', 'forearm.L', 'hand.L', 'palm.04.L', 'little.01.L', 'little.02.L', 'little03.L', 'palm.03.L', 'ring.01.L', 'ring.02.L', 'ring.03.L', 'palm.02.L', 'middle.01.L', 'middle.02.L', 'middle.03.L', 'palm.01.L', 'index.01.L', 'index.02.L', 'index.03.L', 'thumb.01.L', 'thumb.02.L', 'thumb.03.L', 'neck', 'head', 'shoulder.R', 'upperarm.R', 'forearm.R', 'hand.R', 'palm.04.R', 'little.01.R', 'little.02.R', 'little03.R', 'palm.03.R', 'ring.01.R', 'ring.02.R', 'ring.03.R', 'palm.02.R', 'middle.01.R', 'middle.02.R', 'middle.03.R', 'palm.01.R', 'index.01.R', 'index.02.R', 'index.03.R', 'thumb.01.R', 'thumb.02.R', 'thumb.03.R']
            
            self.scale=list(np.linspace(0.4,1.6,500))
            self.translate_x=[i for i in range(-500,500,1)]
            self.translate_y=[i for i in range(-500,500,1)]
            self.rotate=list(np.linspace(-10.0,10.0,num = 500))
            self.no_transform=0
            self.gaussian=list(np.linspace(0.0,5.0,num = 500))
            print("Augemented values len: ", len(self.scale),len(self.translate_x), len(self.translate_y), len(self.rotate), len(self.gaussian))

    def __getitem__(self, i):
        """
        Description: For given row number (i), we find create 32 framed datapoint 
        starting from i (i.e. i to i+32 row). In the process we check if the i and 
        (i+32)th row is from the same video sequence and the last frame is exactly 
        i+32 (since we are removing negative frames, chances are we could be skipping
        some frames in the middle). Once the check is done, we simple loop from i to 
        i+32 row and generate the required dataset.
        """
        sign=self.data['file'][i]
        start = i

        # checking if all the frame from i to i+32 is in the
        # same video sequence or if i+32 would overshoot the
        # dataset length
        for start in range(i, i+num_frames+1):
            if start>=self.dataset_size or self.data['file'][start]!=sign:
                break
        
        # if the previous loop fails, we push start by 
        # a few more frames
        if start!=i+num_frames:
            start=start-(num_frames+1)

        # asserting if first and last frame is 
        # from same video, else, we take a random
        # start frame and start over
        try:
            assert self.data['file'][start]==sign
        except:
            if i-1>0 and i+num_frames<=self.dataset_size:
                i-=1
            elif i+1+num_frames<=self.dataset_size:
                i+=1
            else:
                i=random.randint(100, self.dataset_size-500)
            return self.__getitem__(i)

        # If last frame number is greater than dataset
        # size, we take a random start frame and start over
        if start+num_frames>=self.dataset_size:
            i=random.randint(100, self.dataset_size-500)
            return self.__getitem__(i)

        value=[]
        x=[]
        # If the last frame is not equal to start frame + 32,
        # we take a random start frame and start over
        if self.data['frame'][start]!=self.data['frame'][i]+32:
            i=random.randint(100, self.dataset_size-500)
            return self.__getitem__(i)

        # Additional checking if start and end frame are from the same video
        # If so, the program is terminated
        if self.data['file'][i]!=self.data['file'][i+num_frames]:
            print(self.data['file'][i], self.data['file'][i+num_frames], i, i+num_frames)
            exit(0)
        
        # Initialize the loop variable and all the
        # data augmentation variable
        count=i
        selection_flags = np.random.choice([0, 1], size=(4,), p=[1./2, 1./2])
        temp_scale_x=random.choice(self.scale)
        temp_scale_y=random.choice(self.scale)
        temp_translate_x=random.choice(self.translate_x)
        temp_translate_y=random.choice(self.translate_y)
        temp_rotate=random.choice(self.rotate)
        temp_gaussian=random.choice(self.gaussian)
        noise = np.random.normal(0, temp_gaussian, (num_input_joints,input_dof))

        # Loop from i to i+32 and build the input and output
        # with random data augmentation
        while count!=i+num_frames:
            pose=np.array(self.data['2d_pose'][count]).reshape(67,2)
            input_poses=np.vstack([pose[1:8],pose[25:]])

            if self.no_transform==0:
                # Transformation - Scaling
                if selection_flags[0]==1:
                    input_poses[:,0]*=temp_scale_x
                    input_poses[:,1]*=temp_scale_y

                # Transformation - Translating
                if selection_flags[1]==1:
                    input_poses[:,0]+=temp_translate_x
                    input_poses[:,1]+=temp_translate_y

                # Transformation - Rotation
                if selection_flags[2]==1:
                    theta = np.deg2rad(temp_rotate)
                    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
                    input_poses=np.dot(rot, input_poses.T).T

                # Transformation - Gaussian Noise
                if selection_flags[3]==1:
                    input_poses = input_poses + noise

            # If points after transformation overshoot the image, use default points
            if np.amax(input_poses,axis=0)[0]>1080 or np.amax(input_poses,axis=0)[1]>1920 or np.amin(input_poses,axis=0)[0]<0 or np.amin(input_poses,axis=0)[1]<0:
                input_poses=np.array(self.data['2d_pose'][count].copy()).reshape(67,2)
                input_poses=np.vstack([pose[1:8],pose[25:]])
                self.no_transform=1
                count=i
                x=[]
                value=[]

            # Normalize the pose
            input_poses=input_poses/self.norm_2d

            # Assert if the input poses are between 0 and 1 and append to main list
            assert sum(input_poses.flatten()>=0) == sum(input_poses.flatten()<=1) == input_poses.size
            x.append(input_poses)

            temp_value = self.data['rotation_quaternion'][count]
            temp_value = torch.FloatTensor(temp_value)

            # Build the required ground truth output joints from the
            # entire 60 joint output and append to main list
            val_dict=dict(zip(self.joints,temp_value.view(60,4)))
            filtered = dict(zip(self.body, [val_dict[k].detach().tolist() for k in self.body]))
            value.append(list(filtered.values()))

            count+=1

        # Convert the input and output lists into tensors 
        self.no_transform=0
        value=torch.FloatTensor(value)
        x = torch.FloatTensor(x)
        
        # Assert if the input values are between 0 and 1
        # and output values are between -1 and 1
        assert sum((x.flatten()<0) != (x.flatten()>1))==0
        assert sum((value.flatten()<-1) != (value.flatten()>1))==0

        # Prints the video info. Used for evaluation purposes
        if self.data_type=="VAL":
            print(self.data['file'][i].rsplit("/")[-1],self.data['file'][count].rsplit("/")[-1], i, count)
            print("File: ", self.data['file'][i].rsplit("/")[-1])
            print("Start Frame: ", self.data['frame'][i])
            print("End Frame: ", self.data['frame'][count])

        return [x,value] 

    def __len__(self):
        return self.dataset_size-1


def evaluate(generator, discriminator, gen_input, true_data, val_data):
    """
    Description: Evaluates the generator and discriminator on 
    given input loader and generates the evaluation loss and 
    model's output on the given data point.
    """

    i=0
    gen_input = gen_input[i].unsqueeze(0)
    true_data = true_data[i].unsqueeze(0)

    generator.eval()
    discriminator.eval()

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    loss_generator_1 = torch.nn.L1Loss().to(device)
    loss_generator_2 = nn.MSELoss().to(device)

    generated_data = generator(gen_input)
    true_labels = torch.ones((gen_input.shape[0],1)).to(device)

    generator_discriminator_out = discriminator(generated_data)

    output = {}
    count=1
    for frame in generated_data.view(-1,num_frames, num_output_joints, num_dof).squeeze(0):
        temp=dict(zip(val_data.body,frame.view(num_output_joints,4).detach().tolist()))
        output[str(count)]=temp
        count+=1

    filehandler = open("output_files/output_body_resnet_unet.txt", 'wt')
    output = str(output)
    filehandler.write(output)


    generator_loss_1 = loss_generator_1(generated_data.view(-1,num_frames, num_output_joints, num_dof), true_data)
    generator_loss_2 = loss_generator_2(generator_discriminator_out,true_labels)
    loss = generator_loss_1 + generator_loss_2
    print("Evaluation Loss: ", loss.detach().item())

def train(checkpoint=None):
    """
    Description: Trains the model on the given dataset and saves the checkpoint.
    """

    data_name_train="data/train.csv"
    data_name_eval="data/evaluation.csv"
    train_data=PoseDataset(data_name_train)
    val_data=PoseDataset(data_name_eval, "VAL")
    train_loader = DataLoader(
        train_data,
        batch_size=80,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    valid_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    generator = Generator()
    discriminator = Discriminator()

    generator=generator.to(device)
    discriminator=discriminator.to(device)

    if checkpoint:
        generator.load_state_dict(torch.load(checkpoint)['generator'])
        discriminator.load_state_dict(torch.load(checkpoint)['discriminator'])

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.005)

    generator_scheduler = StepLR(generator_optimizer, step_size=45, gamma=0.1)
    discriminator_scheduler = StepLR(discriminator_optimizer, step_size=45, gamma=0.1)

    loss_generator_1 = torch.nn.L1Loss().to(device)
    loss_generator_2 = nn.MSELoss().to(device)
    count=0
    epochs=90
    generator.train()
    discriminator.train()
    print("Starting Training!!")

    for epoch in range(epochs):
        # iterate through batches
        for i, (x, values) in enumerate(train_loader):
            # Train the generator every iteration and discriminator every 10th epoch
            generator_optimizer.zero_grad()
            gen_input=x.to(device)
            generated_data = generator(gen_input)

            true_data = values.to(device)
            true_labels = torch.ones((x.shape[0],1)).to(device)

            generator_discriminator_out = discriminator(generated_data.to(device))
            generator_loss_1 = loss_generator_1(generated_data.view(-1,num_frames, num_output_joints, num_dof), true_data)
            generator_loss_2 = loss_generator_2(generator_discriminator_out,true_labels)
            loss = generator_loss_1 + generator_loss_2

            loss.backward()
            generator_optimizer.step()

            print("Epoch: %d/%d | Training Loss: %f "%(epoch,epochs,loss.detach().item()))

            if count%10==0:
                # Train the discriminator
                discriminator_optimizer.zero_grad()
                true_discriminator_out = discriminator(true_data.view(-1, num_frames, num_output_joints*num_dof))
                true_discriminator_loss = loss_generator_2(true_discriminator_out, true_labels)

                generator_discriminator_out = discriminator(generated_data.detach())
                generator_discriminator_loss = loss_generator_2(generator_discriminator_out,torch.zeros((x.shape[0],1)).to(device))
                
                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2

                discriminator_loss.backward()
                discriminator_optimizer.step()

                count=0
                print("Epoch: %d/%d | Discriminator Loss: %f"%(epoch, epochs, discriminator_loss.detach().item()))
            
            count+=1

        # Evaluate the model after every epoch
        for j, (x_val, values_val) in enumerate(valid_loader):
            evaluate(generator, discriminator, x_val.to(device), values_val.to(device), val_data)
            break
        
        # Save the checkpoint after every epoch
        torch.save({
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()},
        'checkpoints/checkpoint_body_resnet_unet.pth.tar')
        generator_scheduler.step()
        discriminator_scheduler.step()

if __name__=='__main__':
    train()