
class Options: 
    def __init__(self):
        self.actor_id = 1
        self.actor_count = 1
        self.seed = 1122 
        self.n_step = 256
        self.gamma = 0.99 
        self.env = "BreakoutNoFrameskip-v4" 
        self.env_scale = 1
        self.env_clip_rewards = 0
        self.env_frame_stack = 0
        self.env_episode_life = 1 
        self.env_render = False 
        self.max_episode_length = 50000
        self.eps_base = 0.4 
        self.eps_alpha = 1.0
        self.alpha = 0.6 
        self.beta = 0.4
        self.min_priority = 1e-6
        self.replay_buffer_size = 2000000
        self.sample_begin_size = 1000
        self.learning_begin_size = 1000
        self.batch_size = 512 
        self.learning_rate = 6.25e-5 
        self.learner_storage_capacity = 200000
        self.max_norm = 40.0 
        self.target_update_interval = 2500 
        self.publish_param_interval = 25 
        self.save_interval = 5000 
        self.bps_interval = 100 
