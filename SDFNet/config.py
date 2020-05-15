path = dict(
		# Path to rendering directory
		src_dataset_path = '/scr/devlearning/ShapeNet55_HVC_LR',
		# Input image path. Change to 'image_output' if trained with image
		input_image_path = None,
		# Input depth path. Change to None if trained with image only
		input_depth_path = 'depth_NPZ',
		# Input normal path. Change to None if trained with image only
		input_normal_path = 'normal_output',
		# Input silhouette path. Change to None if trained with image only
		input_seg_path = 'segmentation',
		# Path to sdf and pointcloud directory
		src_pt_path = '/data/devlearning/gensdf_data/ShapeNet_sdf_55',
		# Path to json file
		data_split_json_path = '/data/devlearning/gensdf_data/json_files/data_split.json'
			)
data_setting = dict(
		# Image input size
		input_size = 224,
		img_extension = 'png',
		# Whether to train with 1 random view or all views
		random_view = True,
		seq_len = 25,
		# Specify sub-categories if not trained on all categories
		categories = None
		)
training = dict(
		# Path to model output directory
		out_dir = '/data/devlearning/model_output_test/sdf_3dvc',
		# Training minibatch size
		batch_size = 128,
		# Validation minibatch size
		batch_size_eval = 16,
		# Number of training epochs. Set to None to train forever
		num_epochs = None,
		# Save model every save_model_steps epochs
		save_model_step = 50,
		# Perform validation every eval_step epochs
		eval_step = 1,
		# Perform validation on training set every verbose_step epochs.
		# Note: Sampling might be needed as some objects do not have enough sdf values
		verbose_step = 100000,
		# Number of points sub-sampled during training
		num_points = 2048,
		# If model is continued or resumes
		cont = None,
		# Representation, accepts 'sdf' or 'occ'
		rep = 'sdf',
		# Coordinate system during training: '3dvc', '2dvc' or 'oc'
		coord_system = '3dvc'
		)
logging = dict(
		# Logging directory
		log_dir = '/data/devlearning/model_output_test/log',
		# Experiment name
		exp_name = 'sdf_3dvc'
		)
testing = dict(
		# Evaluation task name
		eval_task_name = 'sdf_3dvc',
		# Size of box to perform matching cube on during mesh generation
		box_size = 1.7,
		# Testing minibatch size. Always 1 if generating mesh on the fly
		batch_size_test = 1,
		# Path to selected model to perform evaluation. If set to None, evaluation will be done on default best models. 
		model_selection_path = None
		) 

