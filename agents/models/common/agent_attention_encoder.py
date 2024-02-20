from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import math


class Self_Attention(nn.Module):

    def __init__(self, hidden_size, attention_head_size, attention_type=None):
        super(Self_Attention, self).__init__()

        self.attention_head_size = attention_head_size
        self.attention_type = attention_type

        self.query = nn.Linear(hidden_size, self.attention_head_size)
        self.key = nn.Linear(hidden_size, self.attention_head_size)
        self.value = nn.Linear(hidden_size, self.attention_head_size)

    def transpose_for_time(self, x):
        # A, T, D

        return x

    def transpose_for_agent(self, x):

        # T, A, D
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.attention_type == 'time':
            query_layer = self.transpose_for_time(mixed_query_layer)
            key_layer = self.transpose_for_time(mixed_key_layer)
            value_layer = self.transpose_for_time(mixed_value_layer)
        elif self.attention_type == 'agent':
            query_layer = self.transpose_for_agent(mixed_query_layer)
            key_layer = self.transpose_for_agent(mixed_key_layer)
            value_layer = self.transpose_for_agent(mixed_value_layer)
        else:
            assert False, "do not have this attention type"

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)

        if self.attention_type == 'agent':
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        return context_layer

# robot: 1, T, D
# objects: n, T, D

class AgentAttEncoder(nn.Module):

    def __init__(self,
                 shape_meta: dict,
                 dim_embedding: int
                 ):

        super(AgentAttEncoder, self).__init__()

        obj_embedding = nn.ModuleDict()
        obj_keys = list()

        for key, attr in shape_meta.items():

            obj_keys.append(key)
            obj_dim = attr['shape'][0]

            obj_embedding[key] = nn.Sequential(nn.Linear(obj_dim, dim_embedding),
                                               # nn.LayerNorm(dim_embedding),
                                               nn.ReLU())

        self.shape_meta = shape_meta
        self.obj_keys = obj_keys
        self.obj_embedding = obj_embedding
        self.agent_attention_1 = Self_Attention(dim_embedding, dim_embedding, attention_type='agent')
        self.agent_attention_2 = Self_Attention(dim_embedding, dim_embedding, attention_type='agent')

    # inputs: B, T, D
    def forward(self, inputs):

        robot_dim_in = self.shape_meta['robot_feature']['shape'][0]
        obj_dim_in = self.shape_meta['obj_feature']['shape'][0]

        B, T, D = inputs.size()

        robot_feature = inputs[:, :, :robot_dim_in].unsqueeze(1)

        obj_feature = inputs[:, :, robot_dim_in:]
        num_obj = int(obj_feature.size()[-1] / obj_dim_in)
        obj_feature = obj_feature.view(B, T, num_obj, obj_dim_in)
        obj_feature = obj_feature.permute(0, 2, 1, 3)

        obs_dict = {'robot_feature': robot_feature,
                    'obj_feature': obj_feature}

        # B, A, T, D
        agent_embedding = []

        for key in self.obj_keys:

            obj_n = obs_dict[key]

            obj_embedding = self.obj_embedding[key](obj_n)

            agent_embedding.append(obj_embedding)

        # B, A, T, D
        agent_embedding = torch.cat(agent_embedding, dim=1)

        agent_embedding_1 = self.agent_attention_1(agent_embedding)
        agent_embedding = agent_embedding + agent_embedding_1

        agent_embedding_2 = self.agent_attention_2(agent_embedding)
        agent_embedding = agent_embedding + agent_embedding_2

        # the robot feature is always in the first dimension
        # B, T, D
        return agent_embedding[:, 0, :, :]
