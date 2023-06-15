import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import numpy as np
import torch

from dataset.dataset_skeleton import MyDataset, DatasetOnMemory
    
class DroneDataset(MyDataset):
    """
    Dataset for Neural-fly.
    """

    def __init__(self, input_data, input_label, input_ws):
        super(DroneDataset, self).__init__(batch_size=100, num_workers=4)
        self.input_data = self.__preprocess(input_data)
        self.input_label = self.__preprocess(input_label)
        self.input_ws = input_ws
        self.test_data = {}
        self.input_ind_sets = {}

    def __preprocess(self, dataset):
        """
        Preprocess the dataset.
        :param dataset: dataset to be preprocessed.
        :return: preprocessed dataset.
        """
        for task in dataset.keys():
            dataset_mean = np.mean(dataset[task], axis=0)
            dataset_std = np.std(dataset[task], axis=0)
            dataset[task] = torch.Tensor(dataset[task] - dataset_mean)/dataset_std
        return dataset
    
    def generate_random_inputs(self, total_len, n, seed=None):
        """
        Generate random inputs.
        :param n: number of inputs to generate.
        :return: generated inputs.
        """
        if seed is not None:
            np.random.seed(seed)
        selected_idx = np.random.choice(total_len, n)
        return selected_idx

        
    def get_dataset(self, task_dict, mixed, test=False, seed = None):
        # TODO: task_dict??
        """
        Get dataset for the task.
        :param str task_name: name of the task
        :param bool mixed: whether to mix the data from different tasks.
        :return: dataset for the tasks.
        """
        task_name_list = task_dict.keys()
        if mixed:
            input_data = []
            input_label = []
            input_ws = []
            for task_name in task_name_list:
                w = task_dict[task_name][0]
                n = task_dict[task_name][1]
                if "test" in task_name:
                    if not test:
                        continue
                    self.sampled_test_tasks.update({task_name: w})
                    ori_task_name = task_name[:-5]
                else:
                    if test:
                        continue
                    ori_task_name = task_name
                    self.sampled_train_tasks.update({task_name: w})
                idx = self.generate_random_inputs(self.input_data[ori_task_name].shape[0], n, seed = seed)
                if task_name in self.input_ind_sets:
                    self.input_ind_sets[task_name].extend(idx.tolist())
                        
                else:
                    self.input_ind_sets[task_name] = idx.tolist()
                input_data += self.input_data[ori_task_name][self.input_ind_sets[task_name]]
                input_label += self.input_label[ori_task_name][self.input_ind_sets[task_name]]
                input_ws += torch.tensor(np.tile(self.input_ws[ori_task_name][0],(len(self.input_ind_sets[task_name]),1)))
            # output = DatasetOnMemory(input_data, np.array(input_label).flatten(), input_ws)
            output = DatasetOnMemory(input_data, input_label, input_ws)

        else:
            output = {}
            for task_name in task_name_list:
                w = task_dict[task_name][0]
                if "test" in task_name:
                    self.sampled_test_tasks.update({task_name: w})
                else:
                    if test:
                        continue
                    self.sampled_train_tasks.update({task_name: w})
                n = task_dict[task_name][1]
                idx = self.generate_random_inputs(self.input_data[task_name].shape[0], n)
                output[task_name] = DatasetOnMemory(self.input_data[task_name][idx], 
                                                    self.input_label[task_name][idx], 
                                                    self.input_ws[task_name][idx])

        return output
    
