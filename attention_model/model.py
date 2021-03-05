import torch
from torch import nn
from torch.nn.functional import softmax
import math


class model_a(nn.Module):
    """ 
    # Maps <state, action> features to next state
    # Uses attention to determine whether another agent's state/action pair is necessary to pay attention to
    """
    
    def __init__(self,args):
        super(model_a, self).__init__()
        self.args = args
        
        # Encode state and action
        self.state_embedding = nn.Linear(args.max_number_of_agents * args.observation_dimension , args.embedding_dimension)
        self.action_embedding = nn.Linear(args.action_dimension, args.embedding_dimension)
  
        # Attention
        self.qkv_projection = nn.Linear(args.embedding_dimension * 2, 2 * (3 * args.embedding_dimension))        
        self.attention = nn.MultiheadAttention(args.embedding_dimension * 2, args.n_heads)

        # Each attended encoding outputs the next observation pair (x',y').  Collectively, it is the next state.
        self.predict = nn.Linear(args.embedding_dimension * 2, 2)


    def forward(self, state, action):
        """
        Conducts a forward pass
        :arg    state           [batch size, number of agents in the state, observation dimension]       
        :arg    action          [batch size, number of agents in the state, action dimension]         
        :output prediction      [batch size, number of agents in the state, observation dimension]                     
        """


        batch_size = state.shape[0]

        # Prepare the state encodings
        state = state.reshape(batch_size, 1, -1)
        state_encoding = self.state_embedding(state.float())
        state_encoding = state_encoding.repeat(1,self.args.max_number_of_agents,1)      # Repeat the state for each agent

        # Prepare the action encodings
        action_encoding = self.action_embedding(action.float())

        # Concatenate the state with each action
        x = torch.cat((state_encoding, action_encoding),dim=2)

        # Attention
        query, key, value = self.qkv_projection(x).permute(1,0,2).chunk(3,dim=2)
        x = self.attention(query, key, value)[0].permute(1,0,2)

        # Predict the next state
        x = self.predict(x)
        
        return x


class model_b(nn.Module):
    """ 
    # Maps <observation, action> pairs to predict the next observation
    # Uses attention to determine whether another agent's observation/action pair is necessary to pay attention to
    """

    def __init__(self,args):
        super(model_b, self).__init__()
        self.args = args
        
        # Encode the inputs
        self.observation_embedding = nn.Linear(args.observation_dimension , args.embedding_dimension)
        self.action_embedding = nn.Linear(args.action_dimension, args.embedding_dimension)

        # Attention mechanism
        self.qkv_projection = nn.Linear(args.embedding_dimension * 2, 2 * (3 * args.embedding_dimension))        
        self.attention = nn.MultiheadAttention(args.embedding_dimension * 2, args.n_heads)

        # Each attended observation/action encoding outputs, (x',y') for each observation in view.
        self.predict = nn.Linear(args.embedding_dimension * 2, 2)


    def forward(self, observation, action):
        """
        Conducts a forward pass
        :arg    observation     [batch size, number of agents observed, observation dimension]       
        :arg    action          [batch size, number of agents observed, action dimension]         
        :output prediction      [batch size, number of agents observed, observation dimension]                     
        """
        # Observation, action encoding
        # Uses attention to determine whether another agent's observation/action pair is necessary to pay attention to

        # Prepare the state encodings
        observation_encoding = self.observation_embedding(observation.float())      

        # Prepare the action encodings
        action_encoding = self.action_embedding(action.float())

        # Concatenate the state with each action
        x = torch.cat((observation_encoding, action_encoding),dim=2)

        # Attention
        query, key, value = self.qkv_projection(x).permute(1,0,2).chunk(3,dim=2)
        x = self.attention(query, key, value)[0].permute(1,0,2)

        # Predict the next state
        x = self.predict(x)
        
        return x


class model_c(nn.Module):
    """ 
    # Maps <observation features, action> pairs to next observation
    # Uses attention to determine whether another <observation feature/action> pair is necessary to pay attention to
    """
    
    def __init__(self, args):
        super(model_c, self).__init__()
        self.args = args
        
        self.observation_embedding = nn.Linear(1 , args.embedding_dimension)
        self.action_embedding = nn.Embedding(args.number_of_actions, args.embedding_dimension)

        self.qkv_projection = nn.Linear(args.embedding_dimension * 2, 3 * args.embedding_dimension)
        self.attention = nn.MultiheadAttention(args.embedding_dimension, args.n_heads)

        self.predict = nn.Linear(args.embedding_dimension, 1)


    def forward(self, observation, action):
        """
        Conducts a forward pass
        :arg    observation     [batch size, number of agents, observation dimension]       
        :arg    action          [batch size, number of agents]         
        :output prediction      [batch size, number of agents * observation_dimension, 1]                     
        """
        batch_size = observation.shape[0]

        # Encode the inputs
        observation_encoding = self.observation_embedding(observation.reshape(batch_size,-1,1).float())
        action_encoding = torch.cat(2*[self.action_embedding(action).float()],dim=1)
        
        x = torch.cat((observation_encoding, action_encoding),dim=2)

        # Attention
        query, key, value = self.qkv_projection(x).permute(1,0,2).chunk(3,dim=2)
        x = self.attention(query, key, value)[0].permute(1,0,2)

        x = self.predict(x)
        
        return x


class Feedforward(nn.Module):

    def __init__(self, args):
        super(Feedforward, self).__init__()

        layers_dimension        = [args.hidden_dimension for _ in range(args.hidden_layers)]
        layers_dimension[0]     = (args.observation_input_dimension + args.action_input_dimension) * args.max_number_of_agents
        layers_dimension[-1]    = args.output_dimension

        self.layers = nn.ModuleList([nn.Linear(layers_dimension[i],layers_dimension[i+1]) \
                                            for i in range(args.hidden_layers-1)])

    def forward(self, observation, action):
        batch_size = observation.shape[0]
        observation = observation.reshape(batch_size, -1)
        action = action.reshape(batch_size, -1)
        x = torch.cat((observation,action), axis=1).float()

        for layer in self.layers:
            x = layer(x)
        return x


class test(nn.Module):
    
    def __init__(self,args):
        super(test, self).__init__()
        self.args = args
        
        self.observation_embedding = nn.Linear(1 , args.embedding_dimension)
        self.action_embedding = nn.Linear(1, args.embedding_dimension)

        self.query = nn.Linear(42, args.output_dimension)
        self.key = nn.Linear(args.embedding_dimension, args.embedding_dimension)
        self.value = nn.Linear(args.embedding_dimension, args.embedding_dimension)

        self.attention = nn.Linear(args.embedding_dimension, args.output_dimension)
        self.predict = nn.Linear(args.embedding_dimension, 1)


    def forward(self, observation, action):
        batch_size = observation.shape[0]
        observation_encoding = self.observation_embedding(observation.reshape(batch_size,-1,1).float())
        action_encoding = self.action_embedding(action.reshape(batch_size,-1,1).float())
        
        x = torch.cat((observation_encoding, action_encoding),dim=1)

        query = self.query(x.permute(0,2,1)).permute(0,2,1)
        key = self.key(x)
        value = self.value(x)
        d_k = query.shape[1]

        weights = softmax(torch.bmm(query,key.transpose(1,2)),dim=2) / math.sqrt(d_k)

        x = torch.bmm(weights, value)

        x = self.predict(x)
        
        return x