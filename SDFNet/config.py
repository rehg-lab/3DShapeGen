path = dict(
		src_dataset_path = '/scr/devlearning/ShapeNet55_HVC_LR',
		input_image_path = None,
		input_depth_path = 'depth_NPZ',
		input_normal_path = 'normal_output',
		input_seg_path = 'segmentation',

		src_pt_path = '/data/devlearning/gensdf_data/ShapeNet_sdf_55',
		data_split_json_path = '/data/devlearning/gensdf_data/json_files/data_split.json',
		src_mesh_path = '/data/devlearning/gensdf_data/ShapeNet_sdf_55'
			)
data_setting = dict(
		input_size = 224,
		img_extension = 'png',
		random_view = True,
		seq_len = 25,
		categories = None
		)
training = dict(
		out_dir = '/data/devlearning/model_output_test/sdf_3dvc',
		batch_size = 128,
		batch_size_eval = 16,

		num_epochs = None,
		save_model_step = 50,
		eval_step = 1,
		verbose_step = 100000,
		num_points = 2048,
		cont = None,

		rep = 'sdf',
		coord_system = '3dvc'
		)
logging = dict(
		log_dir = '/data/devlearning/model_output_test/log',
		exp_name = 'sdf_3dvc'
		)
testing = dict(
		eval_task_name = 'sdf_3dvc',
		box_size = 1.7,
		batch_size_test = 1, # Always 1 if generating mesh on the fly
		model_selection_path = None
		) 

