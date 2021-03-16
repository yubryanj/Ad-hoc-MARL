import torch
from torch import nn
from torch.nn.functional import softmax
import math


class model_a(nn.Module):
    """ 
    #   from state, actions to next state.
    #   this is invariant to the number of agents, i.e., the number of actions
    """
    
    def __init__(self,args):
        super(model_a, self).__init__()
        self.args = args
        
        # Encode state and action
        self.state_embedding = nn.Linear(args.state_dimension , args.state_embedding_dimension)
        self.action_embedding = nn.Linear(args.action_dimension, args.action_embedding_dimension)
  
        # Attention
        state_action_embedding_dim = args.state_embedding_dimension + args.action_embedding_dimension
        self.q_projection = nn.Linear(state_action_embedding_dim, state_action_embedding_dim)        
        self.v_projection = nn.Linear(state_action_embedding_dim, state_action_embedding_dim)        
        self.k_projection = nn.Linear(state_action_embedding_dim, state_action_embedding_dim)        
        self.attention = nn.MultiheadAttention(state_action_embedding_dim, args.n_heads)

        self.predict = nn.Linear(state_action_embedding_dim, args.state_dimension)


    def forward(self, state, action):
        """
        Conducts a forward pass
        :arg    state           [batch size, 1, state dimension]       
        :arg    action          [batch size, number of agents in the state, action dimension]         
        :output prediction      [batch size, 1, state dimension]                     
        """

        # Prepare the embeddings
        state_embedding = self.state_embedding(state.float())
        state_embedding = state_embedding.repeat(1, action.shape[1], 1)
        action_embedding = self.action_embedding(action.float())
        state_action_embedding = torch.cat((state_embedding, action_embedding),dim=2)

        # Attention
        query = self.q_projection(state_action_embedding).permute(1,0,2)
        key = self.k_projection(state_action_embedding).permute(1,0,2)
        value = self.v_projection(state_action_embedding).permute(1,0,2)
        
        x = self.attention(query, key, value)[0].permute(1,0,2)[:,0,:]

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
        self.observation_embedding = nn.Linear(args.observation_dimension , args.observation_embedding_dimension)
        self.action_embedding = nn.Linear(args.action_dimension, args.action_embedding_dimension)

        # Multi-head attention
        observation_action_embedding_dim = args.observation_embedding_dimension + args.action_embedding_dimension
        self.q_projection = nn.Linear(observation_action_embedding_dim, observation_action_embedding_dim)        
        self.v_projection = nn.Linear(observation_action_embedding_dim, observation_action_embedding_dim)        
        self.k_projection = nn.Linear(observation_action_embedding_dim, observation_action_embedding_dim)        
        self.attention = nn.MultiheadAttention(observation_action_embedding_dim, args.n_heads)

        self.predict = nn.Linear(observation_action_embedding_dim, args.observation_dimension)


    def forward(self, observation, action):
        """
        Conducts a forward pass
        :arg    observation     [batch size, number of agents observed, observation dimension]       
        :arg    action          [batch size, number of agents observed, action dimension]         
        :output prediction      [batch size, number of agents observed, observation dimension]                     
        """
        # Observation, action embedding
        # Uses attention to determine whether another agent's observation/action pair is necessary to pay attention to

        # Prepare the embeddings
        observation_embedding = self.observation_embedding(observation.float())      
        action_embedding = self.action_embedding(action.float())
        observation_action_embedding = torch.cat((observation_embedding, action_embedding),dim=2)

        # Attention
        query = self.q_projection(observation_action_embedding).permute(1,0,2)
        key = self.k_projection(observation_action_embedding).permute(1,0,2)
        value = self.v_projection(observation_action_embedding).permute(1,0,2)

        x = self.attention(query, key, value)[0].permute(1,0,2)

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
        
        self.observation_embedding = nn.Linear(args.observation_dimension , args.embedding_dimension)
        self.action_embedding = nn.Linear(args.action_dimension, args.embedding_dimension)

        observation_action_embedding_dim = args.embedding_dimension
        self.q_projection = nn.Linear(observation_action_embedding_dim, observation_action_embedding_dim)        
        self.v_projection = nn.Linear(observation_action_embedding_dim, observation_action_embedding_dim)        
        self.k_projection = nn.Linear(observation_action_embedding_dim, observation_action_embedding_dim)        
        self.attention = nn.MultiheadAttention(observation_action_embedding_dim, args.n_heads)

        self.predict = nn.Linear(observation_action_embedding_dim, args.observation_dimension)


    def forward(self, observation, action):
        """
        Conducts a forward pass
        :arg    observation     [batch size, 1, observation dimension]              # Only the agent's position
        :arg    action          [batch size, number of agents, action dimension]    # Action of other agents     
        :output prediction      [batch size, 1, observation dimension]              # Agent's next position
        """

        # Encode the inputs
        observation_embedding = self.observation_embedding(observation.float())
        action_embedding = self.action_embedding(action.float())
        
        observation_action_embedding = torch.cat((observation_embedding, action_embedding),dim=1)

        # Attention
        query = self.q_projection(observation_action_embedding).permute(1,0,2)
        key = self.k_projection(observation_action_embedding).permute(1,0,2)
        value = self.v_projection(observation_action_embedding).permute(1,0,2)

        x = self.attention(query, key, value)[0].permute(1,0,2)[:,0,:]

        x = self.predict(x)
        
        return x


class Feedforward(nn.Module):

    def __init__(self, args):
        super(Feedforward, self).__init__()

        layers_dimension        = [args.hidden_dimension for _ in range(args.hidden_layers)]
        layers_dimension[0]     = (args.observation_dimension + args.action_dimension) * args.max_number_of_agents
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
        observation_embedding = self.observation_embedding(observation.reshape(batch_size,-1,1).float())
        action_embedding = self.action_embedding(action.reshape(batch_size,-1,1).float())
        
        x = torch.cat((observation_embedding, action_embedding),dim=1)

        query = self.query(x.permute(0,2,1)).permute(0,2,1)
        key = self.key(x)
        value = self.value(x)
        d_k = query.shape[1]

        weights = softmax(torch.bmm(query,key.transpose(1,2)),dim=2) / math.sqrt(d_k)

        x = torch.bmm(weights, value)

        x = self.predict(x)
        
        return x