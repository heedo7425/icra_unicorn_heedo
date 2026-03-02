#include "tsdf_detection_ros/tsdf_server.h"

#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>

#include "tsdf_detection_ros/conversions.h"
#include "tsdf_detection_ros/ros_params.h"

namespace voxblox {

TsdfServer::TsdfServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private)
    : TsdfServer(nh, nh_private, getTsdfMapConfigFromRosParam(nh_private),
                 getTsdfIntegratorConfigFromRosParam(nh_private),
                 getMeshIntegratorConfigFromRosParam(nh_private)) {}

TsdfServer::TsdfServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private,
                       const TsdfMap::Config& config,
                       const TsdfIntegratorBase::Config& integrator_config,
                       const MeshIntegratorConfig& mesh_config)
    : nh_(nh),
      nh_private_(nh_private),
      verbose_(true),
      world_frame_("world"),
      icp_corrected_frame_("icp_corrected"),
      pose_corrected_frame_("pose_corrected"),
      max_block_distance_from_body_(std::numeric_limits<FloatingPoint>::max()),
      slice_level_(0.5),
      use_freespace_pointcloud_(false),
      color_map_(new RainbowColorMap()),
      publish_pointclouds_on_update_(false),
      publish_slices_(false),
      publish_pointclouds_(false),
      publish_tsdf_map_(false),
      cache_mesh_(false),
      enable_icp_(false),
      accumulate_icp_corrections_(true),
      pointcloud_queue_size_(1),
      num_subscribers_tsdf_map_(0),
      transformer_(nh, nh_private) {
  getServerConfigFromRosParam(nh_private);

  // Advertise topics.
  surface_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
          "surface_pointcloud", 1, true);
  tsdf_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >("tsdf_pointcloud",
                                                              1, true);
  occupancy_marker_pub_ =
      nh_private_.advertise<visualization_msgs::MarkerArray>("occupied_nodes",
                                                             1, true);
  tsdf_slice_pub_ = nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >(
      "tsdf_slice", 1, true);
  confidence_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "confidence_pointcloud", 1, true);
  confidence_grid_pub_ =
      nh_private_.advertise<nav_msgs::OccupancyGrid>(
          "confidence_grid", 1, true);
  confidence_slice_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "confidence_slice", 1, true);
  confidence_slice_grid_pub_ =
      nh_private_.advertise<nav_msgs::OccupancyGrid>(
          "confidence_slice_grid", 1, true);
  dynamic_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "dynamic_points", 1, true);
  target_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "target_points", 1, true);
  obstacle_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "obstacle_points", 1, true);
  ground_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "ground_points", 1, true);
  tp_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "eval/true_positives", 1, true);
  tn_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "eval/true_negatives", 1, true);
  fp_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "eval/false_positives", 1, true);
  fn_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "eval/false_negatives", 1, true);
  oor_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "eval/out_of_range", 1, true);

  nh_private_.param("pointcloud_queue_size", pointcloud_queue_size_,
                    pointcloud_queue_size_);
  pointcloud_sub_ = nh_.subscribe("pointcloud", pointcloud_queue_size_,
                                  &TsdfServer::insertPointcloud, this);

  // Optional prior map topic (only active when enable_prior_map: true)
  bool enable_prior_map = false;
  nh_private_.param("enable_prior_map", enable_prior_map, enable_prior_map);
  if (enable_prior_map) {
    std::string prior_map_topic;
    nh_private_.param("prior_map_topic", prior_map_topic, prior_map_topic);
    if (!prior_map_topic.empty()) {
      prior_map_sub_ = nh_.subscribe(prior_map_topic, 1,
                                     &TsdfServer::insertPriorMapPointcloud, this);
      ROS_INFO_STREAM("Subscribing to prior map topic: " << prior_map_topic);
    }
  }

  // Prior map → ever-free initialization (subscribe to latched prior_map topic).
  nh_private_.param("enable_prior_map_ever_free", enable_prior_map_ever_free_,
                    enable_prior_map_ever_free_);
  if (enable_prior_map_ever_free_) {
    std::string prior_map_ef_topic = "/glim_ros/prior_map";
    nh_private_.param("prior_map_ever_free_topic", prior_map_ef_topic,
                      prior_map_ef_topic);
    prior_map_ever_free_sub_ = nh_.subscribe(
        prior_map_ef_topic, 1, &TsdfServer::priorMapEverFreeCallback, this);
    ROS_INFO_STREAM("[PriorMapEverFree] Subscribing to: " << prior_map_ef_topic);
  }

  mesh_pub_ = nh_private_.advertise<voxblox_msgs::Mesh>("mesh", 1, true);

  // Publishing/subscribing to a layer from another node (when using this as
  // a library, for example within a planner).
  tsdf_map_pub_ =
      nh_private_.advertise<voxblox_msgs::Layer>("tsdf_map_out", 1, false);
  tsdf_map_sub_ = nh_private_.subscribe("tsdf_map_in", 1,
                                        &TsdfServer::tsdfMapCallback, this);
  nh_private_.param("publish_tsdf_map", publish_tsdf_map_, publish_tsdf_map_);

  if (use_freespace_pointcloud_) {
    // points that are not inside an object, but may also not be on a surface.
    // These will only be used to mark freespace beyond the truncation distance.
    freespace_pointcloud_sub_ =
        nh_.subscribe("freespace_pointcloud", pointcloud_queue_size_,
                      &TsdfServer::insertFreespacePointcloud, this);
  }

  if (enable_icp_) {
    icp_transform_pub_ = nh_private_.advertise<geometry_msgs::TransformStamped>(
        "icp_transform", 1, true);
    nh_private_.param("icp_corrected_frame", icp_corrected_frame_,
                      icp_corrected_frame_);
    nh_private_.param("pose_corrected_frame", pose_corrected_frame_,
                      pose_corrected_frame_);
  }

  // Initialize TSDF Map and integrator.
  tsdf_map_.reset(new TsdfMap(config));

  std::string method("merged");
  nh_private_.param("method", method, method);
  if (method.compare("simple") == 0) {
    tsdf_integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else if (method.compare("merged") == 0) {
    tsdf_integrator_.reset(new MergedTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else if (method.compare("fast") == 0) {
    tsdf_integrator_.reset(new FastTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else {
    tsdf_integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  }

  // EverFreeIntegrator — reads params from "ever_free_integrator/" namespace.
  {
    dynamic::EverFreeIntegrator::Config ef_cfg;
    nh_private_.param("ever_free_integrator/burn_in_period",
                      ef_cfg.burn_in_period, ef_cfg.burn_in_period);
    nh_private_.param("ever_free_integrator/counter_to_reset",
                      ef_cfg.counter_to_reset, ef_cfg.counter_to_reset);
    nh_private_.param("ever_free_integrator/temporal_buffer",
                      ef_cfg.temporal_buffer, ef_cfg.temporal_buffer);
    nh_private_.param("ever_free_integrator/tsdf_occupancy_threshold",
                      ef_cfg.tsdf_occupancy_threshold,
                      ef_cfg.tsdf_occupancy_threshold);
    nh_private_.param("ever_free_integrator/neighbor_connectivity",
                      ef_cfg.neighbor_connectivity, ef_cfg.neighbor_connectivity);
    nh_private_.param("ever_free_integrator/num_threads",
                      ef_cfg.num_threads, ef_cfg.num_threads);
    // shared_ptr aliasing constructor: shares ownership with tsdf_map_ but
    // points to the inner layer (avoids double-delete on a raw pointer).
    std::shared_ptr<Layer<TsdfVoxel>> layer_ptr(
        tsdf_map_, tsdf_map_->getTsdfLayerPtr());
    ever_free_integrator_ = std::make_unique<dynamic::EverFreeIntegrator>(
        ef_cfg, layer_ptr);
    ROS_INFO("[EverFreeIntegrator] burn_in=%d  counter_to_reset=%d  "
             "temporal_buffer=%d  occ_thresh=%.2f",
             ef_cfg.burn_in_period, ef_cfg.counter_to_reset,
             ef_cfg.temporal_buffer, ef_cfg.tsdf_occupancy_threshold);
  }

  // DynamicSegmenter — reads params from "dynamic_segmenter/" namespace.
  nh_private_.param("enable_dynamic_segmenter", enable_dynamic_segmenter_,
                    enable_dynamic_segmenter_);
  {
    dynamic::DynamicSegmenter::Config ds_cfg;
    nh_private_.param("dynamic_segmenter/min_cluster_size",
                      ds_cfg.min_cluster_size, ds_cfg.min_cluster_size);
    nh_private_.param("dynamic_segmenter/max_cluster_size",
                      ds_cfg.max_cluster_size, ds_cfg.max_cluster_size);
    nh_private_.param("dynamic_segmenter/min_extent",
                      ds_cfg.min_extent, ds_cfg.min_extent);
    nh_private_.param("dynamic_segmenter/max_extent",
                      ds_cfg.max_extent, ds_cfg.max_extent);
    nh_private_.param("dynamic_segmenter/neighbor_connectivity",
                      ds_cfg.neighbor_connectivity, ds_cfg.neighbor_connectivity);
    nh_private_.param("dynamic_segmenter/grow_clusters_twice",
                      ds_cfg.grow_clusters_twice, ds_cfg.grow_clusters_twice);
    nh_private_.param("dynamic_segmenter/min_cluster_separation",
                      ds_cfg.min_cluster_separation, ds_cfg.min_cluster_separation);
    nh_private_.param("dynamic_segmenter/num_threads",
                      ds_cfg.num_threads, ds_cfg.num_threads);
    std::shared_ptr<Layer<TsdfVoxel>> layer_ptr(
        tsdf_map_, tsdf_map_->getTsdfLayerPtr());
    dynamic_segmenter_ = std::make_unique<dynamic::DynamicSegmenter>(
        ds_cfg, layer_ptr);
    nh_private_.param("dynamic_segmenter/check_cluster_separation_exact",
                      ds_cfg.check_cluster_separation_exact,
                      ds_cfg.check_cluster_separation_exact);
    ROS_INFO("[DynamicSegmenter] enabled=%d  min_cluster=%d  max_cluster=%d  "
             "min_ext=%.2f  max_ext=%.2f  conn=%d  exact_sep=%d",
             enable_dynamic_segmenter_, ds_cfg.min_cluster_size,
             ds_cfg.max_cluster_size, ds_cfg.min_extent, ds_cfg.max_extent,
             ds_cfg.neighbor_connectivity,
             ds_cfg.check_cluster_separation_exact);
  }

  // Tracking — reads params from "tracking/" namespace.
  {
    dynamic::Tracking::Config tr_cfg;
    nh_private_.param("tracking/min_track_duration",
                      tr_cfg.min_track_duration, tr_cfg.min_track_duration);
    nh_private_.param("tracking/max_tracking_distance",
                      tr_cfg.max_tracking_distance,
                      tr_cfg.max_tracking_distance);
    tracker_ = std::make_unique<dynamic::Tracking>(tr_cfg);
    ROS_INFO("[Tracking] min_track_duration=%d  max_tracking_distance=%.2f",
             tr_cfg.min_track_duration, tr_cfg.max_tracking_distance);
  }

  // Point-based pipeline (DBSCAN + UKF) — optional replacement.
  nh_private_.param("use_point_based_pipeline", use_point_based_pipeline_,
                    use_point_based_pipeline_);
  if (use_point_based_pipeline_) {
    dynamic::DbscanClusterer::Config db_cfg;
    nh_private_.param("dbscan/eps", db_cfg.eps, db_cfg.eps);
    nh_private_.param("dbscan/min_points", db_cfg.min_points,
                      db_cfg.min_points);
    nh_private_.param("dbscan/min_cluster_size", db_cfg.min_cluster_size,
                      db_cfg.min_cluster_size);
    nh_private_.param("dbscan/max_cluster_size", db_cfg.max_cluster_size,
                      db_cfg.max_cluster_size);
    nh_private_.param("dbscan/min_extent", db_cfg.min_extent,
                      db_cfg.min_extent);
    nh_private_.param("dbscan/max_extent", db_cfg.max_extent,
                      db_cfg.max_extent);
    dbscan_clusterer_ = std::make_unique<dynamic::DbscanClusterer>(db_cfg);

    dynamic::UkfTracker::Config ukf_cfg;
    nh_private_.param("ukf/alpha", ukf_cfg.alpha, ukf_cfg.alpha);
    nh_private_.param("ukf/beta", ukf_cfg.beta, ukf_cfg.beta);
    nh_private_.param("ukf/kappa", ukf_cfg.kappa, ukf_cfg.kappa);
    nh_private_.param("ukf/process_noise_pos", ukf_cfg.process_noise_pos,
                      ukf_cfg.process_noise_pos);
    nh_private_.param("ukf/process_noise_vel", ukf_cfg.process_noise_vel,
                      ukf_cfg.process_noise_vel);
    nh_private_.param("ukf/measurement_noise", ukf_cfg.measurement_noise,
                      ukf_cfg.measurement_noise);
    nh_private_.param("ukf/gate_distance", ukf_cfg.gate_distance,
                      ukf_cfg.gate_distance);
    nh_private_.param("ukf/min_hits_to_confirm", ukf_cfg.min_hits_to_confirm,
                      ukf_cfg.min_hits_to_confirm);
    nh_private_.param("ukf/max_misses_to_delete", ukf_cfg.max_misses_to_delete,
                      ukf_cfg.max_misses_to_delete);
    nh_private_.param("ukf/min_track_duration", ukf_cfg.min_track_duration,
                      ukf_cfg.min_track_duration);
    ukf_tracker_ = std::make_unique<dynamic::UkfTracker>(ukf_cfg);
    last_track_time_ = ros::Time(0);

    ROS_INFO("[PointBasedPipeline] DBSCAN eps=%.2f min_pts=%d | "
             "UKF gate=%.2f min_hits=%d",
             db_cfg.eps, db_cfg.min_points, ukf_cfg.gate_distance,
             ukf_cfg.min_hits_to_confirm);
  }

  // Patchwork++ ground segmentation.
  nh_private_.param("enable_ground_segmentation", enable_ground_segmentation_,
                    enable_ground_segmentation_);
  nh_private_.param("patchwork/enable_voxelization", enable_patchwork_voxelization_,
                    enable_patchwork_voxelization_);
  nh_private_.param("patchwork/voxel_size", patchwork_voxel_size_,
                    patchwork_voxel_size_);

  if (enable_ground_segmentation_) {
    patchwork::Params pw_params;
    nh_private_.param("patchwork/sensor_height", pw_params.sensor_height,
                      pw_params.sensor_height);
    nh_private_.param("patchwork/max_range", pw_params.max_range,
                      pw_params.max_range);
    nh_private_.param("patchwork/min_range", pw_params.min_range,
                      pw_params.min_range);
    nh_private_.param("patchwork/num_iter", pw_params.num_iter,
                      pw_params.num_iter);
    nh_private_.param("patchwork/num_lpr", pw_params.num_lpr,
                      pw_params.num_lpr);
    nh_private_.param("patchwork/num_min_pts", pw_params.num_min_pts,
                      pw_params.num_min_pts);
    nh_private_.param("patchwork/th_seeds", pw_params.th_seeds,
                      pw_params.th_seeds);
    nh_private_.param("patchwork/th_dist", pw_params.th_dist,
                      pw_params.th_dist);
    nh_private_.param("patchwork/uprightness_thr", pw_params.uprightness_thr,
                      pw_params.uprightness_thr);
    pw_params.verbose = false;
    pw_params.enable_RNR = false;  // intensity 정보 없으므로 RNR 비활성화
    ground_segmenter_ = std::make_unique<patchwork::PatchWorkpp>(pw_params);
    ROS_INFO("[Patchwork++] sensor_height=%.2f  max_range=%.1f  "
             "min_range=%.1f  th_seeds=%.3f  th_dist=%.3f",
             pw_params.sensor_height, pw_params.max_range,
             pw_params.min_range, pw_params.th_seeds, pw_params.th_dist);
  }

  mesh_layer_.reset(new MeshLayer(tsdf_map_->block_size()));

  mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(
      mesh_config, tsdf_map_->getTsdfLayerPtr(), mesh_layer_.get()));

  icp_.reset(new ICP(getICPConfigFromRosParam(nh_private)));

  // Advertise services.
  generate_mesh_srv_ = nh_private_.advertiseService(
      "generate_mesh", &TsdfServer::generateMeshCallback, this);
  clear_map_srv_ = nh_private_.advertiseService(
      "clear_map", &TsdfServer::clearMapCallback, this);
  save_map_srv_ = nh_private_.advertiseService(
      "save_map", &TsdfServer::saveMapCallback, this);
  load_map_srv_ = nh_private_.advertiseService(
      "load_map", &TsdfServer::loadMapCallback, this);
  publish_pointclouds_srv_ = nh_private_.advertiseService(
      "publish_pointclouds", &TsdfServer::publishPointcloudsCallback, this);
  publish_tsdf_map_srv_ = nh_private_.advertiseService(
      "publish_map", &TsdfServer::publishTsdfMapCallback, this);

  // If set, use a timer to progressively integrate the mesh.
  double update_mesh_every_n_sec = 1.0;
  nh_private_.param("update_mesh_every_n_sec", update_mesh_every_n_sec,
                    update_mesh_every_n_sec);

  if (update_mesh_every_n_sec > 0.0) {
    update_mesh_timer_ =
        nh_private_.createTimer(ros::Duration(update_mesh_every_n_sec),
                                &TsdfServer::updateMeshEvent, this);
  }

  double publish_map_every_n_sec = 1.0;
  nh_private_.param("publish_map_every_n_sec", publish_map_every_n_sec,
                    publish_map_every_n_sec);

  if (publish_map_every_n_sec > 0.0) {
    publish_map_timer_ =
        nh_private_.createTimer(ros::Duration(publish_map_every_n_sec),
                                &TsdfServer::publishMapEvent, this);
  }

}

void TsdfServer::getServerConfigFromRosParam(
    const ros::NodeHandle& nh_private) {
  // Before subscribing, determine minimum time between messages.
  // 0 by default.
  double min_time_between_msgs_sec = 0.0;
  nh_private.param("min_time_between_msgs_sec", min_time_between_msgs_sec,
                   min_time_between_msgs_sec);
  min_time_between_msgs_.fromSec(min_time_between_msgs_sec);

  nh_private.param("max_block_distance_from_body",
                   max_block_distance_from_body_,
                   max_block_distance_from_body_);
  nh_private.param("slice_level", slice_level_, slice_level_);
  nh_private.param("slice_relative_to_robot", slice_relative_to_robot_,
                   slice_relative_to_robot_);
  nh_private.param("pointcloud_min_z", pointcloud_min_z_, pointcloud_min_z_);
  nh_private.param("pointcloud_max_z", pointcloud_max_z_, pointcloud_max_z_);
  nh_private.param("vis_high_conf_free", vis_high_conf_free_, vis_high_conf_free_);
  nh_private.param("vis_low_conf_free", vis_low_conf_free_, vis_low_conf_free_);
  nh_private.param("vis_occupied", vis_occupied_, vis_occupied_);
  nh_private.param("eval_max_range", eval_max_range_, eval_max_range_);
  nh_private.param("world_frame", world_frame_, world_frame_);
  nh_private.param("publish_pointclouds_on_update",
                   publish_pointclouds_on_update_,
                   publish_pointclouds_on_update_);
  nh_private.param("publish_slices", publish_slices_, publish_slices_);
  nh_private.param("publish_pointclouds", publish_pointclouds_,
                   publish_pointclouds_);

  nh_private.param("use_freespace_pointcloud", use_freespace_pointcloud_,
                   use_freespace_pointcloud_);
  nh_private.param("pointcloud_queue_size", pointcloud_queue_size_,
                   pointcloud_queue_size_);
  nh_private.param("enable_icp", enable_icp_, enable_icp_);
  nh_private.param("accumulate_icp_corrections", accumulate_icp_corrections_,
                   accumulate_icp_corrections_);

  nh_private.param("verbose", verbose_, verbose_);

  // Mesh settings.
  nh_private.param("mesh_filename", mesh_filename_, mesh_filename_);
  std::string color_mode("");
  nh_private.param("color_mode", color_mode, color_mode);
  color_mode_ = getColorModeFromString(color_mode);

  // Color map for intensity pointclouds.
  std::string intensity_colormap("rainbow");
  float intensity_max_value = kDefaultMaxIntensity;
  nh_private.param("intensity_colormap", intensity_colormap,
                   intensity_colormap);
  nh_private.param("intensity_max_value", intensity_max_value,
                   intensity_max_value);

  // Default set in constructor.
  if (intensity_colormap == "rainbow") {
    color_map_.reset(new RainbowColorMap());
  } else if (intensity_colormap == "inverse_rainbow") {
    color_map_.reset(new InverseRainbowColorMap());
  } else if (intensity_colormap == "grayscale") {
    color_map_.reset(new GrayscaleColorMap());
  } else if (intensity_colormap == "inverse_grayscale") {
    color_map_.reset(new InverseGrayscaleColorMap());
  } else if (intensity_colormap == "ironbow") {
    color_map_.reset(new IronbowColorMap());
  } else {
    ROS_ERROR_STREAM("Invalid color map: " << intensity_colormap);
  }
  color_map_->setMaxValue(intensity_max_value);
}

void TsdfServer::processPointCloudMessageAndInsert(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
    const Transformation& T_G_C, const bool is_freespace_pointcloud) {
  // Convert the PCL pointcloud into our awesome format.

  // Horrible hack fix to fix color parsing colors in PCL.
  bool color_pointcloud = false;
  bool has_intensity = false;
  for (size_t d = 0; d < pointcloud_msg->fields.size(); ++d) {
    if (pointcloud_msg->fields[d].name == std::string("rgb")) {
      pointcloud_msg->fields[d].datatype = sensor_msgs::PointField::FLOAT32;
      color_pointcloud = true;
    } else if (pointcloud_msg->fields[d].name == std::string("intensity")) {
      has_intensity = true;
    }
  }

  Pointcloud points_C;
  Colors colors;
  timing::Timer ptcloud_timer("ptcloud_preprocess");

  // Convert differently depending on RGB or I type.
  // Always use PointXYZ if no color — avoids "Failed to find match for
  // field 'intensity'" warnings from Livox/custom pointclouds.
  if (color_pointcloud) {
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  } else {
    pcl::PointCloud<pcl::PointXYZ> pointcloud_pcl;
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  }
  ptcloud_timer.Stop();

  Transformation T_G_C_refined = T_G_C;
  if (enable_icp_) {
    timing::Timer icp_timer("icp");
    if (!accumulate_icp_corrections_) {
      icp_corrected_transform_.setIdentity();
    }
    static Transformation T_offset;
    const size_t num_icp_updates =
        icp_->runICP(tsdf_map_->getTsdfLayer(), points_C,
                     icp_corrected_transform_ * T_G_C, &T_G_C_refined);
    if (verbose_) {
      ROS_INFO("ICP refinement performed %zu successful update steps",
               num_icp_updates);
    }
    icp_corrected_transform_ = T_G_C_refined * T_G_C.inverse();

    if (!icp_->refiningRollPitch()) {
      // its already removed internally but small floating point errors can
      // build up if accumulating transforms
      Transformation::Vector6 T_vec = icp_corrected_transform_.log();
      T_vec[3] = 0.0;
      T_vec[4] = 0.0;
      icp_corrected_transform_ = Transformation::exp(T_vec);
    }

    // Publish transforms as both TF and message.
    tf::Transform icp_tf_msg, pose_tf_msg;
    geometry_msgs::TransformStamped transform_msg;

    tf::transformKindrToTF(icp_corrected_transform_.cast<double>(),
                           &icp_tf_msg);
    tf::transformKindrToTF(T_G_C.cast<double>(), &pose_tf_msg);
    tf::transformKindrToMsg(icp_corrected_transform_.cast<double>(),
                            &transform_msg.transform);
    tf_broadcaster_.sendTransform(
        tf::StampedTransform(icp_tf_msg, pointcloud_msg->header.stamp,
                             world_frame_, icp_corrected_frame_));
    tf_broadcaster_.sendTransform(
        tf::StampedTransform(pose_tf_msg, pointcloud_msg->header.stamp,
                             icp_corrected_frame_, pose_corrected_frame_));

    transform_msg.header.frame_id = world_frame_;
    transform_msg.child_frame_id = icp_corrected_frame_;
    icp_transform_pub_.publish(transform_msg);

    icp_timer.Stop();
  }

  // Store sensor Z for robot-relative slice.
  current_sensor_z_ = T_G_C_refined.getPosition().z();

  // ---------------------------------------------------------------------------
  // Pipeline order matching dynablox:
  //   1. Segment  (uses PREVIOUS frame's TSDF distances + ever-free labels)
  //   2. Ever-free update  (uses PREVIOUS frame's TSDF distances)
  //   3. TSDF integration  (updates distances/weights with current scan)
  // ---------------------------------------------------------------------------

  ++frame_counter_;

  // 0a. Voxelization (Downsampling) — sensor frame, BEFORE world transform.
  if (enable_patchwork_voxelization_ && !points_C.empty() && !is_freespace_pointcloud) {
    timing::Timer voxel_timer("patchwork_voxelization");
    const size_t orig_size = points_C.size();

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl_cloud->resize(points_C.size());
    for (size_t i = 0; i < points_C.size(); ++i) {
      (*pcl_cloud)[i].x = points_C[i].x();
      (*pcl_cloud)[i].y = points_C[i].y();
      (*pcl_cloud)[i].z = points_C[i].z();
    }

    pcl::VoxelGrid<pcl::PointXYZ> vox;
    vox.setInputCloud(pcl_cloud);
    const float leaf = static_cast<float>(patchwork_voxel_size_);
    vox.setLeafSize(leaf, leaf, leaf);
    pcl::PointCloud<pcl::PointXYZ> down;
    vox.filter(down);

    points_C.resize(down.size());
    for (size_t i = 0; i < down.size(); ++i) {
      points_C[i] = Point(down[i].x, down[i].y, down[i].z);
    }
    // Colors are lost after voxelization; reset to uniform.
    colors.assign(points_C.size(), Color(128, 128, 128));

    voxel_timer.Stop();
    ROS_INFO_THROTTLE(2.0, "[Voxelization] %lu -> %lu points", orig_size, points_C.size());
  }

  // 0b. World transform + Z filter in one pass (single transform).
  const bool need_world_pts =
      (enable_dynamic_segmenter_ || enable_ground_segmentation_) && !is_freespace_pointcloud;
  const bool need_z_filter =
      pointcloud_min_z_ > -1e10 || pointcloud_max_z_ < 1e10;

  Pointcloud points_W;
  if (need_world_pts || need_z_filter) {
    if (need_z_filter) {
      Pointcloud filtered_C;
      Colors filtered_cols;
      filtered_C.reserve(points_C.size());
      filtered_cols.reserve(colors.size());
      if (need_world_pts) points_W.reserve(points_C.size());

      for (size_t i = 0; i < points_C.size(); ++i) {
        const Point pw = T_G_C_refined * points_C[i].cast<FloatingPoint>();
        if (pw.z() >= pointcloud_min_z_ && pw.z() <= pointcloud_max_z_) {
          filtered_C.push_back(points_C[i]);
          if (i < colors.size()) filtered_cols.push_back(colors[i]);
          if (need_world_pts) points_W.push_back(pw);
        }
      }
      points_C = std::move(filtered_C);
      colors = std::move(filtered_cols);
    } else {
      points_W.resize(points_C.size());
      for (size_t i = 0; i < points_C.size(); ++i) {
        points_W[i] = T_G_C_refined * points_C[i].cast<FloatingPoint>();
      }
    }
  }

  // 0c. Ground segmentation (Patchwork++) — sensor frame.
  std::vector<bool> is_ground(points_C.size(), false);
  if (enable_ground_segmentation_ && !is_freespace_pointcloud) {
    timing::Timer gnd_timer("ground_segmentation");
    // Zero-copy: Eigen::Vector3f is 3 contiguous floats, so Pointcloud data
    // is laid out as [x0,y0,z0, x1,y1,z1, ...] — row-major Nx3.
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>
        cloud_eigen(points_C[0].data(), points_C.size(), 3);
    ground_segmenter_->estimateGround(cloud_eigen);
    const Eigen::VectorXi ground_idx = ground_segmenter_->getGroundIndices();
    for (int i = 0; i < ground_idx.size(); ++i) {
      const int idx = ground_idx(i);
      if (idx >= 0 && idx < static_cast<int>(is_ground.size())) {
        is_ground[idx] = true;
      }
    }
    gnd_timer.Stop();

    // Publish ground / obstacle (non-ground) points.
    const bool pub_obs = obstacle_pointcloud_pub_.getNumSubscribers() > 0;
    const bool pub_gnd = ground_pointcloud_pub_.getNumSubscribers() > 0;
    if ((pub_obs || pub_gnd) && !points_W.empty()) {
      pcl::PointCloud<pcl::PointXYZRGB> obs_cloud, gnd_cloud;
      obs_cloud.header.frame_id = world_frame_;
      gnd_cloud.header.frame_id = world_frame_;
      for (size_t i = 0; i < points_W.size(); ++i) {
        pcl::PointXYZRGB pt;
        pt.x = points_W[i].x();
        pt.y = points_W[i].y();
        pt.z = points_W[i].z();
        if (is_ground[i]) {
          if (pub_gnd) {
            pt.r = 100; pt.g = 200; pt.b = 100;  // 연두
            gnd_cloud.push_back(pt);
          }
        } else {
          if (pub_obs) {
            pt.r = 255; pt.g = 140; pt.b = 0;  // orange
            obs_cloud.push_back(pt);
          }
        }
      }
      if (pub_obs) obstacle_pointcloud_pub_.publish(obs_cloud);
      if (pub_gnd) ground_pointcloud_pub_.publish(gnd_cloud);
    }

    ROS_INFO_THROTTLE(2.0, "[Patchwork++] ground=%ld  nonground=%ld  total=%lu",
                      static_cast<long>(ground_idx.size()),
                      static_cast<long>(points_C.size()) - ground_idx.size(),
                      points_C.size());
  }

  // 1. Dynamic segmentation — uses previous frame's map state.
  //    If ground segmentation is on, only feed non-ground world points.
  std::vector<bool> is_dynamic;
  std::vector<bool> ever_free_flags;
  if (enable_dynamic_segmenter_ && !is_freespace_pointcloud) {
    // Build non-ground world points with index mapping.
    Pointcloud seg_input;
    std::vector<size_t> seg_to_orig;  // seg_input index → original index
    seg_input.reserve(points_W.size());
    seg_to_orig.reserve(points_W.size());
    for (size_t i = 0; i < points_W.size(); ++i) {
      if (enable_ground_segmentation_ && is_ground[i]) continue;
      seg_to_orig.push_back(i);
      seg_input.push_back(points_W[i]);
    }

    timing::Timer seg_timer("dynamic_segmentation");
    std::vector<bool> seg_dynamic =
        dynamic_segmenter_->segment(seg_input, frame_counter_, &ever_free_flags);
    seg_timer.Stop();

    // Clustering + Tracking.
    std::vector<dynamic::Cluster> clusters;
    if (use_point_based_pipeline_ && dbscan_clusterer_ && ukf_tracker_) {
      // Point-based: DBSCAN on ever-free points + UKF tracking.
      timing::Timer dbscan_timer("dbscan_clustering");
      clusters = dbscan_clusterer_->cluster(seg_input, ever_free_flags);
      dbscan_timer.Stop();

      timing::Timer ukf_timer("ukf_tracking");
      float dt = 0.1f;
      if (last_track_time_.toSec() > 0) {
        dt = static_cast<float>(
            (pointcloud_msg->header.stamp - last_track_time_).toSec());
      }
      last_track_time_ = pointcloud_msg->header.stamp;
      ukf_tracker_->track(seg_input, clusters, dt);
      ukf_timer.Stop();
    } else {
      // Voxel-based: flood-fill + centroid matching.
      timing::Timer track_timer("tracking");
      clusters = dynamic_segmenter_->lastClusters();
      tracker_->track(seg_input, clusters);
      track_timer.Stop();
    }

    std::fill(seg_dynamic.begin(), seg_dynamic.end(), false);
    // Per-point cluster ID for color-coded visualization (-1 = not dynamic).
    std::vector<int> point_cluster_id(seg_dynamic.size(), -1);
    int valid_count = 0;
    for (const auto& c : clusters) {
      if (!c.valid) continue;
      ++valid_count;
      for (size_t idx : c.point_indices) {
        seg_dynamic[idx] = true;
        point_cluster_id[idx] = c.id;
      }
    }

    ROS_INFO_THROTTLE(2.0,
        "[Tracking] frame=%d  seg_input=%lu  clusters=%lu  valid=%d  pipeline=%s",
        frame_counter_, seg_input.size(), clusters.size(), valid_count,
        use_point_based_pipeline_ ? "DBSCAN+UKF" : "VoxelFF+Centroid");

    // Map back to original indices (always indexed via seg_to_orig).
    is_dynamic.resize(points_W.size(), false);
    last_point_cluster_id_.assign(points_W.size(), -1);
    for (size_t i = 0; i < seg_dynamic.size(); ++i) {
      if (seg_dynamic[i]) {
        is_dynamic[seg_to_orig[i]] = true;
        last_point_cluster_id_[seg_to_orig[i]] = point_cluster_id[i];
      }
    }
    std::vector<bool> ef_full(points_W.size(), false);
    for (size_t i = 0; i < ever_free_flags.size(); ++i) {
      if (ever_free_flags[i]) {
        ef_full[seg_to_orig[i]] = true;
      }
    }
    ever_free_flags = std::move(ef_full);
  }

  // 2. Ever-free update — uses previous frame's TSDF distances.
  ever_free_integrator_->updateEverFreeVoxels(frame_counter_);

  // 3. Publish dynamic points and evaluation metrics.
  if (!is_dynamic.empty()) {
    const Point sensor_pos = T_G_C_refined.getPosition();
    const double max_range_sq = eval_max_range_ * eval_max_range_;

    const bool pub_dyn = dynamic_pointcloud_pub_.getNumSubscribers() > 0;
    const bool pub_tgt = target_pointcloud_pub_.getNumSubscribers() > 0;
    const bool pub_eval = tp_pointcloud_pub_.getNumSubscribers() > 0 ||
                          fp_pointcloud_pub_.getNumSubscribers() > 0 ||
                          fn_pointcloud_pub_.getNumSubscribers() > 0 ||
                          tn_pointcloud_pub_.getNumSubscribers() > 0 ||
                          oor_pointcloud_pub_.getNumSubscribers() > 0;

    pcl::PointCloud<pcl::PointXYZRGB> dyn_cloud, tgt_cloud, tp_cloud,
        tn_cloud, fp_cloud, fn_cloud, oor_cloud;
    dyn_cloud.header.frame_id = world_frame_;
    tgt_cloud.header.frame_id = world_frame_;

    if (pub_eval) {
      tp_cloud.header.frame_id  = world_frame_;
      tn_cloud.header.frame_id  = world_frame_;
      fp_cloud.header.frame_id  = world_frame_;
      fn_cloud.header.frame_id  = world_frame_;
      oor_cloud.header.frame_id = world_frame_;
    }

    for (size_t i = 0; i < points_W.size(); ++i) {
      // Skip ground points entirely.
      if (i < is_ground.size() && is_ground[i]) continue;

      const bool is_dyn = is_dynamic[i];
      const bool ef = (i < ever_free_flags.size()) && ever_free_flags[i];

      // Only compute range for points that need it (dynamic or ever-free).
      const bool in_range = (is_dyn || ef)
          ? (points_W[i] - sensor_pos).squaredNorm() <= max_range_sq
          : true;

      // Eval metrics — skip TN (majority) when no subscriber for it.
      if (pub_eval && (is_dyn || ef ||
                       oor_pointcloud_pub_.getNumSubscribers() > 0 ||
                       tn_pointcloud_pub_.getNumSubscribers() > 0)) {
        pcl::PointXYZRGB pt;
        pt.x = points_W[i].x();
        pt.y = points_W[i].y();
        pt.z = points_W[i].z();

        if (!in_range) {
          pt.r = 128; pt.g = 128; pt.b = 128;
          oor_cloud.push_back(pt);
        } else if (is_dyn && ef) {
          pt.r = 0; pt.g = 255; pt.b = 0;       // TP green
          tp_cloud.push_back(pt);
        } else if (is_dyn && !ef) {
          pt.r = 0; pt.g = 100; pt.b = 255;     // FP blue
          fp_cloud.push_back(pt);
        } else if (!is_dyn && ef) {
          pt.r = 255; pt.g = 0; pt.b = 0;       // FN red
          fn_cloud.push_back(pt);
        } else {
          pt.r = 0; pt.g = 0; pt.b = 0;         // TN black
          tn_cloud.push_back(pt);
        }
      }

      // Dynamic points = TP + FP only (same as eval, colored by cluster ID).
      if (is_dyn && in_range && pub_dyn) {
        pcl::PointXYZRGB pt;
        pt.x = points_W[i].x();
        pt.y = points_W[i].y();
        pt.z = points_W[i].z();
        const int cid = (i < last_point_cluster_id_.size())
                             ? last_point_cluster_id_[i] : 0;
        const uint8_t h = static_cast<uint8_t>(cid * 67 + 31);
        pt.r = static_cast<uint8_t>((h * 23) % 200 + 55);
        pt.g = static_cast<uint8_t>((h * 47) % 200 + 55);
        pt.b = static_cast<uint8_t>((h * 71) % 200 + 55);
        dyn_cloud.push_back(pt);
      }

      // Target points = TP + FN (ever_free voxels occupied by current scan).
      if (ef && in_range && pub_tgt) {
        pcl::PointXYZRGB pt;
        pt.x = points_W[i].x();
        pt.y = points_W[i].y();
        pt.z = points_W[i].z();
        pt.r = 255; pt.g = 0; pt.b = 255;  // magenta
        tgt_cloud.push_back(pt);
      }
    }

    if (pub_dyn) dynamic_pointcloud_pub_.publish(dyn_cloud);
    if (pub_tgt) target_pointcloud_pub_.publish(tgt_cloud);
    if (pub_eval) {
      tp_pointcloud_pub_.publish(tp_cloud);
      tn_pointcloud_pub_.publish(tn_cloud);
      fp_pointcloud_pub_.publish(fp_cloud);
      fn_pointcloud_pub_.publish(fn_cloud);
      oor_pointcloud_pub_.publish(oor_cloud);
    }

    ROS_INFO_THROTTLE(2.0,
        "[Eval] frame=%d  TP=%lu  FP=%lu  FN=%lu  TN=%lu  OOR=%lu  dyn=%lu  tgt=%lu",
        frame_counter_, tp_cloud.size(), fp_cloud.size(),
        fn_cloud.size(), tn_cloud.size(), oor_cloud.size(),
        dyn_cloud.size(), tgt_cloud.size());
  }

  // 3. TSDF integration — now update the map with the current scan.
  if (verbose_) {
    ROS_INFO("Integrating a pointcloud with %lu points.", points_C.size());
  }

  ros::WallTime start = ros::WallTime::now();
  integratePointcloud(T_G_C_refined, points_C, colors, is_freespace_pointcloud);
  ros::WallTime end = ros::WallTime::now();
  if (verbose_) {
    ROS_INFO("Finished integrating in %f seconds, have %lu blocks.",
             (end - start).toSec(),
             tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks());
  }

  timing::Timer block_remove_timer("remove_distant_blocks");
  tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_block_distance_from_body_);
  mesh_layer_->clearDistantMesh(T_G_C.getPosition(),
                                max_block_distance_from_body_);
  block_remove_timer.Stop();

  // Callback for inheriting classes.
  newPoseCallback(T_G_C);
}

// Checks if we can get the next message from queue.
bool TsdfServer::getNextPointcloudFromQueue(
    std::queue<sensor_msgs::PointCloud2::Ptr>* queue,
    sensor_msgs::PointCloud2::Ptr* pointcloud_msg, Transformation* T_G_C) {
  const size_t kMaxQueueSize = 10;
  if (queue->empty()) {
    return false;
  }
  *pointcloud_msg = queue->front();
  if (transformer_.lookupTransform((*pointcloud_msg)->header.frame_id,
                                   world_frame_,
                                   (*pointcloud_msg)->header.stamp, T_G_C)) {
    queue->pop();
    return true;
  } else {
    if (queue->size() >= kMaxQueueSize) {
      ROS_ERROR_THROTTLE(60,
                         "Input pointcloud queue getting too long! Dropping "
                         "some pointclouds. Either unable to look up transform "
                         "timestamps or the processing is taking too long.");
      while (queue->size() >= kMaxQueueSize) {
        queue->pop();
      }
    }
  }
  return false;
}

void TsdfServer::insertPointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg_in) {
  if (pointcloud_msg_in->header.stamp - last_msg_time_ptcloud_ >
      min_time_between_msgs_) {
    last_msg_time_ptcloud_ = pointcloud_msg_in->header.stamp;
    // So we have to process the queue anyway... Push this back.
    pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::PointCloud2::Ptr pointcloud_msg;
  bool processed_any = false;
  while (
      getNextPointcloudFromQueue(&pointcloud_queue_, &pointcloud_msg, &T_G_C)) {
    constexpr bool is_freespace_pointcloud = false;
    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
    processed_any = true;
  }

  if (!processed_any) {
    return;
  }

  if (publish_pointclouds_on_update_) {
    publishPointclouds();
  }

  if (verbose_) {
    ROS_INFO_STREAM("Timings: " << std::endl << timing::Timing::Print());
    ROS_INFO_STREAM(
        "Layer memory: " << tsdf_map_->getTsdfLayer().getMemorySize());
  }
}

void TsdfServer::insertFreespacePointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg_in) {
  if (pointcloud_msg_in->header.stamp - last_msg_time_freespace_ptcloud_ >
      min_time_between_msgs_) {
    last_msg_time_freespace_ptcloud_ = pointcloud_msg_in->header.stamp;
    // So we have to process the queue anyway... Push this back.
    freespace_pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::PointCloud2::Ptr pointcloud_msg;
  while (getNextPointcloudFromQueue(&freespace_pointcloud_queue_,
                                    &pointcloud_msg, &T_G_C)) {
    constexpr bool is_freespace_pointcloud = true;
    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
  }
}

void TsdfServer::insertPriorMapPointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg) {
  // Prior map is already in world frame: use identity transform (T_G_C = I)
  Transformation T_G_C;  // defaults to identity
  ROS_INFO("Integrating prior map pointcloud with identity transform.");
  constexpr bool is_freespace_pointcloud = false;
  processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                    is_freespace_pointcloud);
  publishPointclouds();
}

void TsdfServer::priorMapEverFreeCallback(
    const sensor_msgs::PointCloud2::ConstPtr& msg) {
  // One-shot: unsubscribe after first reception (latched topic may re-deliver).
  prior_map_ever_free_sub_.shutdown();

  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*msg, cloud);

  const float voxel_size = tsdf_map_->getTsdfLayer().voxel_size();
  const float voxel_size_inv = 1.0f / voxel_size;

  prior_occupied_voxels_.clear();
  for (const auto& pt : cloud) {
    if (pt.z < pointcloud_min_z_ || pt.z > pointcloud_max_z_) continue;
    const Point p(pt.x, pt.y, pt.z);
    const GlobalIndex gidx = getGridIndexFromPoint<GlobalIndex>(p, voxel_size_inv);
    prior_occupied_voxels_.insert(gidx);
  }

  prior_map_ever_free_loaded_ = true;

  // Pass the set to the ever-free integrator.
  ever_free_integrator_->setPriorOccupiedVoxels(&prior_occupied_voxels_);

  ROS_INFO("[PriorMapEverFree] Loaded %lu points, %lu occupied voxels",
           cloud.size(), prior_occupied_voxels_.size());
}

void TsdfServer::integratePointcloud(const Transformation& T_G_C,
                                     const Pointcloud& ptcloud_C,
                                     const Colors& colors,
                                     const bool is_freespace_pointcloud) {
  CHECK_EQ(ptcloud_C.size(), colors.size());
  tsdf_integrator_->integratePointCloud(T_G_C, ptcloud_C, colors,
                                        is_freespace_pointcloud);
}

void TsdfServer::publishAllUpdatedTsdfVoxels() {
  // Create a pointcloud with distance = intensity.
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  // Publish a single Z slice (plane_index 2 = Z axis).
  const double eff_slice = slice_relative_to_robot_
      ? current_sensor_z_ + slice_level_ : slice_level_;
  createDistancePointcloudFromTsdfLayerSlice(
      tsdf_map_->getTsdfLayer(), 2u,
      static_cast<FloatingPoint>(eff_slice), &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  tsdf_pointcloud_pub_.publish(pointcloud);
}

void TsdfServer::publishEverFreeConfidencePointcloud() {
  if (confidence_pointcloud_pub_.getNumSubscribers() == 0) return;

  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud.header.frame_id = world_frame_;

  // Use the same occupancy threshold as EverFreeIntegrator.
  const float occ_thresh = ever_free_integrator_
      ? ever_free_integrator_->getConfig().tsdf_occupancy_threshold
      : 0.3f;

  BlockIndexList all_blocks;
  tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&all_blocks);

  for (const BlockIndex& bidx : all_blocks) {
    const Block<TsdfVoxel>& block =
        tsdf_map_->getTsdfLayer().getBlockByIndex(bidx);

    for (size_t i = 0; i < block.num_voxels(); ++i) {
      const TsdfVoxel& vox = block.getVoxelByLinearIndex(i);

      // Skip unobserved voxels.
      if (vox.weight < 1e-6f) continue;

      // Voxel centre in world frame.
      const VoxelIndex vidx = block.computeVoxelIndexFromLinearIndex(i);
      const Point centre = block.computeCoordinatesFromVoxelIndex(vidx);

      pcl::PointXYZRGB pt;
      pt.x = centre.x();
      pt.y = centre.y();
      pt.z = centre.z();

      if (vox.distance < occ_thresh) {
        if (!vis_occupied_) continue;
        pt.r = 220; pt.g = 60;  pt.b = 60;
      } else if (vox.ever_free) {
        if (!vis_high_conf_free_) continue;
        pt.r = 0;   pt.g = 100; pt.b = 255;
      } else {
        if (!vis_low_conf_free_) continue;
        pt.r = 0;   pt.g = 200; pt.b = 180;
      }

      cloud.push_back(pt);
    }
  }

  confidence_pointcloud_pub_.publish(cloud);
}

void TsdfServer::publishEverFreeConfidence2DGrid() {
  if (confidence_grid_pub_.getNumSubscribers() == 0) return;

  BlockIndexList all_blocks;
  tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&all_blocks);
  if (all_blocks.empty()) return;

  const float vs         = tsdf_map_->getTsdfLayer().voxel_size();
  const float block_size = tsdf_map_->getTsdfLayer().block_size();
  const float occ_thresh = 0.3f;

  // Z-range for projection: keep voxels between these heights.
  float z_min_proj = 0.1f;
  float z_max_proj = 0.3f;
  nh_private_.param("grid_z_min", z_min_proj, z_min_proj);
  nh_private_.param("grid_z_max", z_max_proj, z_max_proj);

  // Find XY bounding box of all allocated blocks.
  float x_min =  std::numeric_limits<float>::max();
  float x_max =  std::numeric_limits<float>::lowest();
  float y_min =  std::numeric_limits<float>::max();
  float y_max =  std::numeric_limits<float>::lowest();

  for (const BlockIndex& bidx : all_blocks) {
    const Point origin =
        tsdf_map_->getTsdfLayer().getBlockByIndex(bidx).origin();
    x_min = std::min(x_min, origin.x());
    x_max = std::max(x_max, origin.x() + block_size);
    y_min = std::min(y_min, origin.y());
    y_max = std::max(y_max, origin.y() + block_size);
  }

  const int nx = static_cast<int>(std::ceil((x_max - x_min) / vs));
  const int ny = static_cast<int>(std::ceil((y_max - y_min) / vs));
  if (nx <= 0 || ny <= 0) return;

  nav_msgs::OccupancyGrid grid;
  grid.header.frame_id    = world_frame_;
  grid.header.stamp       = ros::Time::now();
  grid.info.resolution    = vs;
  grid.info.width         = static_cast<uint32_t>(nx);
  grid.info.height        = static_cast<uint32_t>(ny);
  grid.info.origin.position.x  = x_min;
  grid.info.origin.position.y  = y_min;
  grid.info.origin.position.z  = 0.0;
  grid.info.origin.orientation.w = 1.0;
  grid.data.assign(static_cast<size_t>(nx * ny), -1);  // -1 = unknown

  // Priority per cell: occupied(100) > high-conf-free(0) > low-conf-free(50)
  // We track the highest-priority value seen so far.
  auto priority = [](int8_t v) -> int {
    if (v == 100) return 3;
    if (v ==   0) return 2;
    if (v ==  50) return 1;
    return 0;  // unknown (-1)
  };

  for (const BlockIndex& bidx : all_blocks) {
    const Block<TsdfVoxel>& block =
        tsdf_map_->getTsdfLayer().getBlockByIndex(bidx);

    for (size_t i = 0; i < block.num_voxels(); ++i) {
      const TsdfVoxel& vox = block.getVoxelByLinearIndex(i);
      if (vox.weight < 1e-6f) continue;

      const VoxelIndex vidx   = block.computeVoxelIndexFromLinearIndex(i);
      const Point      centre = block.computeCoordinatesFromVoxelIndex(vidx);

      if (centre.z() < z_min_proj || centre.z() > z_max_proj) continue;

      const int gx = static_cast<int>((centre.x() - x_min) / vs);
      const int gy = static_cast<int>((centre.y() - y_min) / vs);
      if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) continue;

      int8_t new_val;
      if (vox.distance < occ_thresh) {
        if (!vis_occupied_) continue;
        new_val = 100;  // occupied
      } else if (vox.ever_free) {
        if (!vis_high_conf_free_) continue;
        new_val = 0;    // high-confidence free
      } else {
        if (!vis_low_conf_free_) continue;
        new_val = 50;   // low-confidence free
      }

      const size_t cell = static_cast<size_t>(gy * nx + gx);
      if (priority(new_val) > priority(grid.data[cell])) {
        grid.data[cell] = new_val;
      }
    }
  }

  confidence_grid_pub_.publish(grid);
}

void TsdfServer::publishEverFreeConfidenceSlice() {
  if (confidence_slice_pub_.getNumSubscribers() == 0) return;

  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud.header.frame_id = world_frame_;

  const float occ_thresh = 0.3f;
  const float vs = tsdf_map_->getTsdfLayer().voxel_size();
  const float half_vs = vs * 0.5f;
  const double eff_slice = slice_relative_to_robot_
      ? current_sensor_z_ + slice_level_ : slice_level_;
  const float z_level = static_cast<float>(eff_slice);

  BlockIndexList all_blocks;
  tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&all_blocks);

  for (const BlockIndex& bidx : all_blocks) {
    const Block<TsdfVoxel>& block =
        tsdf_map_->getTsdfLayer().getBlockByIndex(bidx);

    for (size_t i = 0; i < block.num_voxels(); ++i) {
      const TsdfVoxel& vox = block.getVoxelByLinearIndex(i);
      if (vox.weight < 1e-6f) continue;

      const VoxelIndex vidx = block.computeVoxelIndexFromLinearIndex(i);
      const Point centre = block.computeCoordinatesFromVoxelIndex(vidx);

      if (std::abs(centre.z() - z_level) > half_vs) continue;

      pcl::PointXYZRGB pt;
      pt.x = centre.x();
      pt.y = centre.y();
      pt.z = centre.z();

      if (vox.distance < occ_thresh) {
        if (!vis_occupied_) continue;
        pt.r = 220; pt.g = 60;  pt.b = 60;
      } else if (vox.ever_free) {
        if (!vis_high_conf_free_) continue;
        pt.r = 0;   pt.g = 100; pt.b = 255;
      } else {
        if (!vis_low_conf_free_) continue;
        pt.r = 0;   pt.g = 200; pt.b = 180;
      }
      cloud.push_back(pt);
    }
  }
  confidence_slice_pub_.publish(cloud);
}

void TsdfServer::publishEverFreeConfidenceSlice2DGrid() {
  if (confidence_slice_grid_pub_.getNumSubscribers() == 0) return;

  BlockIndexList all_blocks;
  tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&all_blocks);
  if (all_blocks.empty()) return;

  const float vs         = tsdf_map_->getTsdfLayer().voxel_size();
  const float block_size = tsdf_map_->getTsdfLayer().block_size();
  const float occ_thresh = 0.3f;
  const float half_vs    = vs * 0.5f;
  const double eff_slice = slice_relative_to_robot_
      ? current_sensor_z_ + slice_level_ : slice_level_;
  const float z_level    = static_cast<float>(eff_slice);

  float x_min =  std::numeric_limits<float>::max();
  float x_max =  std::numeric_limits<float>::lowest();
  float y_min =  std::numeric_limits<float>::max();
  float y_max =  std::numeric_limits<float>::lowest();

  for (const BlockIndex& bidx : all_blocks) {
    const Point origin =
        tsdf_map_->getTsdfLayer().getBlockByIndex(bidx).origin();
    x_min = std::min(x_min, origin.x());
    x_max = std::max(x_max, origin.x() + block_size);
    y_min = std::min(y_min, origin.y());
    y_max = std::max(y_max, origin.y() + block_size);
  }

  const int nx = static_cast<int>(std::ceil((x_max - x_min) / vs));
  const int ny = static_cast<int>(std::ceil((y_max - y_min) / vs));
  if (nx <= 0 || ny <= 0) return;

  nav_msgs::OccupancyGrid grid;
  grid.header.frame_id    = world_frame_;
  grid.header.stamp       = ros::Time::now();
  grid.info.resolution    = vs;
  grid.info.width         = static_cast<uint32_t>(nx);
  grid.info.height        = static_cast<uint32_t>(ny);
  grid.info.origin.position.x  = x_min;
  grid.info.origin.position.y  = y_min;
  grid.info.origin.position.z  = 0.0;
  grid.info.origin.orientation.w = 1.0;
  grid.data.assign(static_cast<size_t>(nx * ny), -1);

  auto priority = [](int8_t v) -> int {
    if (v == 100) return 3;
    if (v ==   0) return 2;
    if (v ==  50) return 1;
    return 0;
  };

  for (const BlockIndex& bidx : all_blocks) {
    const Block<TsdfVoxel>& block =
        tsdf_map_->getTsdfLayer().getBlockByIndex(bidx);

    for (size_t i = 0; i < block.num_voxels(); ++i) {
      const TsdfVoxel& vox = block.getVoxelByLinearIndex(i);
      if (vox.weight < 1e-6f) continue;

      const VoxelIndex vidx   = block.computeVoxelIndexFromLinearIndex(i);
      const Point      centre = block.computeCoordinatesFromVoxelIndex(vidx);

      if (std::abs(centre.z() - z_level) > half_vs) continue;

      const int gx = static_cast<int>((centre.x() - x_min) / vs);
      const int gy = static_cast<int>((centre.y() - y_min) / vs);
      if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) continue;

      int8_t new_val;
      if (vox.distance < occ_thresh) {
        if (!vis_occupied_) continue;
        new_val = 100;
      } else if (vox.ever_free) {
        if (!vis_high_conf_free_) continue;
        new_val = 0;
      } else {
        if (!vis_low_conf_free_) continue;
        new_val = 50;
      }

      const size_t cell = static_cast<size_t>(gy * nx + gx);
      if (priority(new_val) > priority(grid.data[cell])) {
        grid.data[cell] = new_val;
      }
    }
  }
  confidence_slice_grid_pub_.publish(grid);
}

void TsdfServer::publishTsdfSurfacePoints() {
  // Create a pointcloud with distance = intensity.
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  const float surface_distance_thresh =
      tsdf_map_->getTsdfLayer().voxel_size() * 0.75;
  createSurfacePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(),
                                       surface_distance_thresh, &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  surface_pointcloud_pub_.publish(pointcloud);
}

void TsdfServer::publishTsdfOccupiedNodes() {
  // Create a pointcloud with distance = intensity.
  visualization_msgs::MarkerArray marker_array;
  createOccupancyBlocksFromTsdfLayer(tsdf_map_->getTsdfLayer(), world_frame_,
                                     &marker_array);
  occupancy_marker_pub_.publish(marker_array);
}

void TsdfServer::publishSlices() {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  const double eff_slice = slice_relative_to_robot_
      ? current_sensor_z_ + slice_level_ : slice_level_;
  createDistancePointcloudFromTsdfLayerSlice(tsdf_map_->getTsdfLayer(), 2,
                                             eff_slice, &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  tsdf_slice_pub_.publish(pointcloud);
}

void TsdfServer::publishMap(bool reset_remote_map) {
  if (!publish_tsdf_map_) {
    return;
  }
  int subscribers = this->tsdf_map_pub_.getNumSubscribers();
  if (subscribers > 0) {
    if (num_subscribers_tsdf_map_ < subscribers) {
      // Always reset the remote map and send all when a new subscriber
      // subscribes. A bit of overhead for other subscribers, but better than
      // inconsistent map states.
      reset_remote_map = true;
    }
    const bool only_updated = !reset_remote_map;
    timing::Timer publish_map_timer("map/publish_tsdf");
    voxblox_msgs::Layer layer_msg;
    serializeLayerAsMsg<TsdfVoxel>(this->tsdf_map_->getTsdfLayer(),
                                   only_updated, &layer_msg);
    if (reset_remote_map) {
      layer_msg.action = static_cast<uint8_t>(MapDerializationAction::kReset);
    }
    this->tsdf_map_pub_.publish(layer_msg);
    publish_map_timer.Stop();
  }
  num_subscribers_tsdf_map_ = subscribers;
}

void TsdfServer::publishPointclouds() {
  // Only publish if someone is listening — these traverse the full map.
  if (tsdf_pointcloud_pub_.getNumSubscribers() > 0) {
    publishAllUpdatedTsdfVoxels();
  }
  if (surface_pointcloud_pub_.getNumSubscribers() > 0) {
    publishTsdfSurfacePoints();
  }
  if (occupancy_marker_pub_.getNumSubscribers() > 0) {
    publishTsdfOccupiedNodes();
  }
  if (publish_slices_ && tsdf_slice_pub_.getNumSubscribers() > 0) {
    publishSlices();
  }
}

void TsdfServer::updateMesh() {
  if (verbose_) {
    ROS_INFO("Updating mesh.");
  }

  timing::Timer generate_mesh_timer("mesh/update");
  constexpr bool only_mesh_updated_blocks = true;
  constexpr bool clear_updated_flag = true;
  mesh_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");

  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  if (cache_mesh_) {
    cached_mesh_msg_ = mesh_msg;
  }

  publish_mesh_timer.Stop();

  if (publish_pointclouds_ && !publish_pointclouds_on_update_) {
    publishPointclouds();
  }
}

bool TsdfServer::generateMesh() {
  timing::Timer generate_mesh_timer("mesh/generate");
  const bool clear_mesh = true;
  if (clear_mesh) {
    constexpr bool only_mesh_updated_blocks = false;
    constexpr bool clear_updated_flag = true;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  } else {
    constexpr bool only_mesh_updated_blocks = true;
    constexpr bool clear_updated_flag = true;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  }
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");
  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  publish_mesh_timer.Stop();

  if (!mesh_filename_.empty()) {
    timing::Timer output_mesh_timer("mesh/output");
    const bool success = outputMeshLayerAsPly(mesh_filename_, *mesh_layer_);
    output_mesh_timer.Stop();
    if (success) {
      ROS_INFO("Output file as PLY: %s", mesh_filename_.c_str());
    } else {
      ROS_INFO("Failed to output mesh as PLY: %s", mesh_filename_.c_str());
    }
  }

  ROS_INFO_STREAM("Mesh Timings: " << std::endl << timing::Timing::Print());
  return true;
}

bool TsdfServer::saveMap(const std::string& file_path) {
  // Inheriting classes should add saving other layers to this function.
  return io::SaveLayer(tsdf_map_->getTsdfLayer(), file_path);
}

bool TsdfServer::loadMap(const std::string& file_path) {
  // Inheriting classes should add other layers to load, as this will only
  // load
  // the TSDF layer.
  constexpr bool kMulitpleLayerSupport = true;
  bool success = io::LoadBlocksFromFile(
      file_path, Layer<TsdfVoxel>::BlockMergingStrategy::kReplace,
      kMulitpleLayerSupport, tsdf_map_->getTsdfLayerPtr());
  if (success) {
    LOG(INFO) << "Successfully loaded TSDF layer.";
  }
  return success;
}

bool TsdfServer::clearMapCallback(std_srvs::Empty::Request& /*request*/,
                                  std_srvs::Empty::Response&
                                  /*response*/) {  // NOLINT
  clear();
  return true;
}

bool TsdfServer::generateMeshCallback(std_srvs::Empty::Request& /*request*/,
                                      std_srvs::Empty::Response&
                                      /*response*/) {  // NOLINT
  return generateMesh();
}

bool TsdfServer::saveMapCallback(voxblox_msgs::FilePath::Request& request,
                                 voxblox_msgs::FilePath::Response&
                                 /*response*/) {  // NOLINT
  return saveMap(request.file_path);
}

bool TsdfServer::loadMapCallback(voxblox_msgs::FilePath::Request& request,
                                 voxblox_msgs::FilePath::Response&
                                 /*response*/) {  // NOLINT
  bool success = loadMap(request.file_path);
  return success;
}

bool TsdfServer::publishPointcloudsCallback(
    std_srvs::Empty::Request& /*request*/, std_srvs::Empty::Response&
    /*response*/) {  // NOLINT
  publishPointclouds();
  return true;
}

bool TsdfServer::publishTsdfMapCallback(std_srvs::Empty::Request& /*request*/,
                                        std_srvs::Empty::Response&
                                        /*response*/) {  // NOLINT
  publishMap();
  return true;
}

void TsdfServer::updateMeshEvent(const ros::TimerEvent& /*event*/) {
  updateMesh();
}

void TsdfServer::publishMapEvent(const ros::TimerEvent& /*event*/) {
  // Heavy confidence visualizations run here (timer-based, not per-scan).
  publishEverFreeConfidencePointcloud();
  publishEverFreeConfidence2DGrid();
  publishEverFreeConfidenceSlice();
  publishEverFreeConfidenceSlice2DGrid();
  publishMap();
}

void TsdfServer::clear() {
  tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
  mesh_layer_->clear();

  // Publish a message to reset the map to all subscribers.
  if (publish_tsdf_map_) {
    constexpr bool kResetRemoteMap = true;
    publishMap(kResetRemoteMap);
  }
}

void TsdfServer::tsdfMapCallback(const voxblox_msgs::Layer& layer_msg) {
  timing::Timer receive_map_timer("map/receive_tsdf");

  bool success =
      deserializeMsgToLayer<TsdfVoxel>(layer_msg, tsdf_map_->getTsdfLayerPtr());

  if (!success) {
    ROS_ERROR_THROTTLE(10, "Got an invalid TSDF map message!");
  } else {
    ROS_INFO_ONCE("Got an TSDF map from ROS topic!");
    if (publish_pointclouds_on_update_) {
      publishPointclouds();
    }
  }
}

}  // namespace voxblox
