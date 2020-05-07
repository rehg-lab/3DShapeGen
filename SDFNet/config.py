path = dict(
		# src_dataset_path = '/media/ant/5072f36a-8c4f-445d-89f5-577b5b1e05d9/ShapeNet_13_RGB_D+N',
		# src_dataset_path = '/scr/devlearning/ShapeNet55_HVC_LR',
		# src_dataset_path = '/scr/devlearning/ShapeNet55_EVC_LR',
		# src_dataset_path = '/data/devlearning/ShapeNet55_EVC_LR_PRED_UNSEEN/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_LR_PRED/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_LR_PRED_UNSEEN/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_LR_PRED_SEEN_TEST',

		
		# src_dataset_path = '/data/devlearning/ShapeNet55_EVC_LR_PRED_SEEN_TEST/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_basic/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_basic_PRED_SEEN/',
		src_dataset_path = '/data/devlearning/LRBG_study_PRED/HVC_basic_100_samples_LRB',



		########### Variability
		# src_dataset_path = '/data/devlearning/data_for_LRBg_variability_study/ShapeNet55_HVC_light_VAL_100_samples',
		# src_dataset_path = '/data/devlearning/data_for_LRBg_variability_study/ShapeNet55_HVC_reflectance+light_VAL_100_samples',
		# src_dataset_path = '/data/devlearning/data_for_LRBg_variability_study/ShapeNet55_HVC_reflectance+light+background_VAL_100_samples',




		# src_dataset_path = '/data/devlearning/ShapeNet55_EVC_LR_PRED_2ND_backup',


		# input_image_path = 'image_output',
		input_image_path = None,

		# input_depth_path = 'depth_output',
		# input_depth_path = 'depth_NPZ',
		input_depth_path = 'depth_pred_NPZ',

		# input_depth_path = None,


		# input_normal_path = 'normal_output',
		input_normal_path = 'normals_pred_output',

		# input_normal_path = None,

		input_seg_path = 'segmentation',
		# input_seg_path = None,


		# src_pt_path = '/media/ant/50b6d91a-50d7-45a7-b77c-92b79a62e56a/data/ShapeNet.build',
		src_pt_path = '/data/devlearning/gensdf_data/ShapeNet_sdf_55',	
		input_points_path = '',
		input_pointcloud_path = '',
		input_metadata_path = '',
		# data_split_json_path = '/data/devlearning/gensdf_data/json_files/data_13_42_split_unseen.json',
		# data_split_json_path = '/data/devlearning/gensdf_data/json_files/data_split.json',
		######### For LRBg
		data_split_json_path = '/data/devlearning/gensdf_data/json_files/val_LRBg_study_updated.json',
		src_mesh_path = '/data/devlearning/gensdf_data/ShapeNet_sdf_55'
			)
data_setting = dict(
		input_size = 224,
		img_extension = 'png',
		random_view = True,
		seq_len = 25,
		# categories = ['03001627']
		# categories = ['03636649']

		# categories = ['02691156','02828884','02933112','02958343','03001627','03211117','03636649','03691459','04090263','04256520','04379243','04401088','04530566']
		categories = None
		)
training = dict(
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_gt_dn_hvc',
		# out_dir = '/data/devlearning/model_output_bmvc/img_hvc',
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_pred_dn_evc',
		# out_dir = '/data/devlearning/model_output_bmvc/occnet_evc',
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_gt_dn_oc',
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_pred_dn_hvc',

		# out_dir = '/data/devlearning/model_output_bmvc/sdf_img_hvc_basic',
		out_dir = '/data/devlearning/model_output_bmvc/sdf_pred_dn_hvc_basic',







		# batch_size = 128,
		batch_size = 256,

		# batch_size_eval = 16,
		batch_size_eval = 32,

		num_epochs = None,
		save_model_step = 50,
		eval_step = 1,
		verbose_step = 100000,
		num_points = 2048,
		cont = None,
		# cont = 'model-900.pth.tar',
		# algo = 'occnet',

		rep = 'sdf',
		# coord_system = 'vc'
		coord_system = 'hvc'
		# coord_system = 'oc'


		)
logging = dict(
		log_dir = '/scr/devlearning/log',
		# exp_name = 'sdf_gt_dn_hvc' # Change
		# exp_name = 'img_hvc' # Change
		# exp_name = 'sdf_pred_dn_evc' # Change
		# exp_name = 'sdf_img_hvc_basic' # Change


		# exp_name = 'occnet_evc' # Change
		# exp_name = 'sdf_gt_dn_oc' # Change

		# exp_name = 'sdf_pred_dn_hvc' # Change

		exp_name = 'sdf_pred_dn_hvc_basic' # Change







		)
testing = dict(
		# eval_task_name = 'sdf_gt_dn_hvc_unseen_3deg', #Change
		# eval_task_name = 'sdf_gt_dn_hvc_unseen_2deg', #Change
		# eval_task_name = 'sdf_gt_dn_hvc_seen_2deg', #Change
		# eval_task_name = 'sdf_gt_dn_hvc_seen_3deg', #Change
		# eval_task_name = 'sdf_gt_dn_hvc_basic_val', #Change
		

		# eval_task_name = 'sdf_gt_dn_oc_unseen_3deg', #Change




		# eval_task_name = 'occnet_evc', #Change
		# eval_task_name = 'occnet_evc_unseen_2deg', #Change
		# eval_task_name = 'occnet_evc_seen_2deg', #Change




		# eval_task_name = 'img_hvc', #Change
		# eval_task_name = 'img_hvc_unseen_3deg', #Change
		# eval_task_name = 'img_hvc_seen_3deg', #Change


		# eval_task_name = 'sdf_pred_dn_evc', #Change
		# eval_task_name = 'sdf_pred_dn_evc_unseen_2deg', #Change
		# eval_task_name = 'sdf_pred_dn_evc_seen_2deg', #Change


		# eval_task_name = 'sdf_pred_dn_hvc', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_unseen_3deg', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_seen_3deg', #Change



		# eval_task_name = 'img_hvc_basic', #Change
		# eval_task_name = 'img_hvc_basic_val', #Change

		# eval_task_name = 'img_hvc_basic_light_val', #Change
		# eval_task_name = 'img_hvc_basic_lr_val', #Change
		# eval_task_name = 'img_hvc_basic_lrbg_val', #Change

		# eval_task_name = 'sdf_pred_dn_hvc_basic', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_basic_val', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_basic_light_val', #Change
		eval_task_name = 'sdf_pred_dn_hvc_basic_LRB_val', #Change











		box_size = 1.7,
		# box_size = 1.01,

		batch_size_test = 1,
		model_selection_path = None
		) 

