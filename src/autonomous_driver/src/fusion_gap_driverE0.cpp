// fusion_driver.cpp (ROS 2 Humble)
// Racing upgrades implemented:
//  1) Speed-aware obstacle inflation "bubble" in FTG bins (distance-dependent angular blocking)
//  2) Best-point-in-gap target selection (scores bins inside the chosen gap, not just center)
//  3) TTC-based braking + yaw clamp as TTC shrinks
//  4) Adaptive FOV (wide on open straights, tight near walls/clutter)
//  5) Better FTG/wall blending based on turn demand + corridor asymmetry + proximity
//  6) Input filtering: Z-gate, temporal range smoothing with "release", and median filter across bins
//  7) Control smoothing: yaw-rate slew limiter; optional curvature filtering via Kalman1D
//  8) Perf: preallocated vectors, no per-scan copies, throttled logs

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <limits>
#include <cmath>
#include <mutex>
#include <vector>
#include <algorithm>
#include <tuple>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "autonomous_driver/pid.hpp"
#include "autonomous_driver/kalman_1d.hpp"

class FusionDriver : public rclcpp::Node
{
public:
  FusionDriver()
  : Node("fusion_driver"),
    wall_pid_(0.6, 0.08, 0.25),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    RCLCPP_INFO(this->get_logger(), "FusionDriver node started");

    // ---- Frame/sign params ----
    base_frame_     = this->declare_parameter<std::string>("base_frame", "base_link");
    forward_sign_   = this->declare_parameter<int>("forward_sign", 1);
    lateral_sign_   = this->declare_parameter<int>("lateral_sign", 1);
    cmd_w_sign_     = this->declare_parameter<int>("cmd_w_sign", 1);

    // ---- Behavior params ----
    max_speed_      = this->declare_parameter<double>("max_speed", 6.0);
    min_speed_      = this->declare_parameter<double>("min_speed", 0.8);

    // Base yaw clamp (still used as absolute ceiling)
    max_yaw_rate_   = this->declare_parameter<double>("max_yaw_rate", 1.5);

    heading_gain_   = this->declare_parameter<double>("heading_gain", 2.0);

    // FTG config
    gap_threshold_  = this->declare_parameter<double>("gap_threshold", 1.0);
    fov_tight_deg_  = this->declare_parameter<double>("ftg_fov_tight_deg", 35.0);
    fov_wide_deg_   = this->declare_parameter<double>("ftg_fov_wide_deg", 70.0);
    self_radius_    = this->declare_parameter<double>("self_radius", 0.40);

    // Obstacle inflation bubble
    bubble_margin_base_ = this->declare_parameter<double>("bubble_margin_base", 0.08);  // m
    bubble_margin_kv_   = this->declare_parameter<double>("bubble_margin_kv", 0.03);    // m per (m/s)
    bubble_max_bins_    = this->declare_parameter<int>("bubble_max_bins", 25);

    // FTG gap scoring weights
    score_w_range_      = this->declare_parameter<double>("score_w_range", 1.0);
    score_w_center_     = this->declare_parameter<double>("score_w_center", 0.20);
    score_w_edge_       = this->declare_parameter<double>("score_w_edge", 0.15);
    gap_edge_margin_bins_= this->declare_parameter<int>("gap_edge_margin_bins", 2);

    // Stops (distance)
    near_stop_center_ = this->declare_parameter<double>("near_stop_center", 0.7);
    near_stop_wide_   = this->declare_parameter<double>("near_stop_wide", 0.6);
    wall_stop_        = this->declare_parameter<double>("wall_stop", 0.35);

    // TTC braking
    ttc_hard_stop_   = this->declare_parameter<double>("ttc_hard_stop", 0.55); // s
    ttc_soft_start_  = this->declare_parameter<double>("ttc_soft_start", 1.20); // s (start reducing speed/yaw)
    ttc_yaw_scale_   = this->declare_parameter<double>("ttc_yaw_scale", 0.65); // clamp yaw as TTC shrinks

    // Speed slew
    accel_up_        = this->declare_parameter<double>("accel_up", 3.0);   // m/s^2
    accel_down_      = this->declare_parameter<double>("accel_down", 6.0); // m/s^2

    // Yaw slew (jerk-ish limiting)
    yaw_accel_limit_ = this->declare_parameter<double>("yaw_accel_limit", 6.0); // rad/s^2

    // Speed-dependent yaw clamp
    yaw_min_rate_    = this->declare_parameter<double>("yaw_min_rate", 0.35); // rad/s
    yaw_vref_        = this->declare_parameter<double>("yaw_vref", 3.0);      // m/s (above this clamp tightens)

    // Lat-accel based curve speed
    a_lat_max_       = this->declare_parameter<double>("a_lat_max", 3.0);   // m/s^2

    // Range filtering
    z_min_           = this->declare_parameter<double>("z_min", -0.35);
    z_max_           = this->declare_parameter<double>("z_max", +0.90);
    range_ema_alpha_ = this->declare_parameter<double>("range_ema_alpha", 0.35);
    range_release_mps_= this->declare_parameter<double>("range_release_mps", 8.0); // how fast ranges "relax" upward
    median_window_   = this->declare_parameter<int>("median_window", 5); // odd: 3/5/7

    // Blending tuning
    blend_turn_bias_ = this->declare_parameter<double>("blend_turn_bias", 0.85); // FTG dominance in corners
    blend_straight_bias_ = this->declare_parameter<double>("blend_straight_bias", 0.35); // FTG dominance on straights
    asym_gain_       = this->declare_parameter<double>("corridor_asym_gain", 0.30);

    // Curvature filtering (recommended for racing)
    use_curvature_filter_ = this->declare_parameter<bool>("use_curvature_filter", true);

    cmd_pub_    = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("steering_arc", 10);

    auto sensor_qos = rclcpp::SensorDataQoS();

    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "top_3d_lidar_plugin/out", sensor_qos,
      std::bind(&FusionDriver::lidarCallback, this, std::placeholders::_1));

    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "front_rgbd/depth/image_raw", sensor_qos,
      std::bind(&FusionDriver::depthCallback, this, std::placeholders::_1));

    // Preallocate bin vectors (fixed bins: [-90, +90], 1 deg resolution)
    min_ranges_.assign(kNumBins, inf());
    ema_ranges_.assign(kNumBins, inf());
    work_ranges_.assign(kNumBins, inf());
    median_ranges_.assign(kNumBins, inf());

    RCLCPP_INFO(this->get_logger(),
      "Params: base_frame=%s forward_sign=%d lateral_sign=%d cmd_w_sign=%d",
      base_frame_.c_str(), forward_sign_, lateral_sign_, cmd_w_sign_);
  }

private:
  // ---------------- Const binning ----------------
  static constexpr int    kNumBins   = 180;
  static constexpr double kAngleRes  = 1.0 * M_PI / 180.0; // rad/bin
  static constexpr double kHalfFov   = M_PI / 2.0;
  static constexpr int    kForwardBin= kNumBins / 2;

  // ---------------- Helpers ----------------
  static inline double inf() { return std::numeric_limits<double>::infinity(); }

  static inline double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  }

  static inline int clampi(int x, int lo, int hi) {
    return std::max(lo, std::min(hi, x));
  }

  static inline double lerp(double a, double b, double t) {
    return a + (b - a) * t;
  }

  static inline double approach(double current, double target, double max_delta) {
    if (target > current) return std::min(target, current + max_delta);
    return std::max(target, current - max_delta);
  }

  static inline bool finite(double x) { return std::isfinite(x); }

  static inline double binToAngle(int bin) {
    // bins cover [-pi/2, +pi/2]
    return (static_cast<double>(bin) + 0.5) * kAngleRes - M_PI / 2.0;
  }

  static inline int angleToBin(double angle) {
    double t = (angle + M_PI / 2.0) / kAngleRes;
    return static_cast<int>(std::floor(t));
  }

  // ---------------- Follow-The-Gap bins ----------------
  struct GapBins {
    int start = 0;
    int end   = 0;
    int size() const { return end - start + 1; }
  };

  static std::vector<GapBins> findGapsInBins(const std::vector<double>& ranges,
                                             int i_min, int i_max,
                                             double gap_threshold)
  {
    std::vector<GapBins> gaps;
    bool in_gap = false;
    int start = i_min;

    for (int i = i_min; i <= i_max; ++i) {
      bool free_bin = finite(ranges[i]) && (ranges[i] > gap_threshold);
      if (free_bin && !in_gap) {
        in_gap = true;
        start = i;
      } else if (!free_bin && in_gap) {
        in_gap = false;
        gaps.push_back(GapBins{start, i - 1});
      }
    }
    if (in_gap) gaps.push_back(GapBins{start, i_max});
    return gaps;
  }

  // Pick best gap with center bias (your original idea), then pick best point INSIDE the gap by scoring bins
  std::tuple<bool, double, int, int, double, double>
  followTheGapOnceScored(const std::vector<double>& ranges_eff,
                         int i_min, int i_max,
                         double gap_threshold,
                         double used_max_range,
                         double theta_goal,
                         double alpha)
  {
    // Nearest obstacle distance inside window
    double d_min = inf();
    for (int i = i_min; i <= i_max; ++i) {
      if (!finite(ranges_eff[i])) continue;
      d_min = std::min(d_min, ranges_eff[i]);
    }

    // Find gaps
    std::vector<GapBins> gaps = findGapsInBins(ranges_eff, i_min, i_max, gap_threshold);
    if (gaps.empty()) {
      return {false, 0.0, -1, -1, 0.0, d_min};
    }

    // Best gap selection: large AND near forward
    const double center_bias = 0.35;
    auto best_it = std::max_element(
      gaps.begin(), gaps.end(),
      [&](const GapBins& a, const GapBins& b){
        double ca = 0.5 * (a.start + a.end);
        double cb = 0.5 * (b.start + b.end);
        double sa = static_cast<double>(a.size()) - center_bias * std::abs(ca - kForwardBin);
        double sb = static_cast<double>(b.size()) - center_bias * std::abs(cb - kForwardBin);
        return sa < sb;
      });

    GapBins best_gap = *best_it;

    // Score bins inside best gap
    int s = best_gap.start;
    int e = best_gap.end;

    // Avoid gap edges unless gap is tiny
    int s_in = s;
    int e_in = e;
    if ((e - s + 1) > 2 * gap_edge_margin_bins_) {
      s_in = s + gap_edge_margin_bins_;
      e_in = e - gap_edge_margin_bins_;
    }

    int best_bin = clampi((s + e) / 2, i_min, i_max);
    double best_score = -1e18;

    for (int i = s_in; i <= e_in; ++i) {
      double r = ranges_eff[i];
      double r_use = finite(r) ? r : used_max_range;

      // prefer farther range
      double term_range = score_w_range_ * r_use;

      // small penalty for steering away from forward
      double term_center = -score_w_center_ * std::abs(i - kForwardBin);

      // edge penalty: prefer middle of gap so we don't graze walls at high speed
      double gap_center = 0.5 * (s + e);
      double term_edge = -score_w_edge_ * std::abs(i - gap_center);

      double score = term_range + term_center + term_edge;
      if (score > best_score) {
        best_score = score;
        best_bin = i;
      }
    }

    double theta_c = binToAngle(best_bin);

    // Original "final heading" blending of goal and gap direction weighted by d_min
    double dm = d_min;
    if (!finite(dm) || dm <= 1e-3) dm = 1e-3;
    double w = alpha / dm;
    double theta_final = (w * theta_c + theta_goal) / (w + 1.0);

    if (!finite(theta_final)) {
      return {false, 0.0, -1, -1, theta_c, d_min};
    }

    return {true, theta_final, s, e, theta_c, d_min};
  }

  // Adaptive FTG: tries different max-range cappings, but now uses precomputed eff ranges (no copying)
  std::tuple<bool, double, int, int, double, double, double>
  followTheGapAdaptiveScored(const std::vector<double>& ranges_eff,
                             int i_min, int i_max,
                             double gap_threshold)
  {
    const double theta_goal = 0.0;
    const double alpha      = 2.5;

    bool ok = false;
    double theta_final = 0.0;
    double theta_c = 0.0;
    double d_min = inf();
    int gap_s = -1, gap_e = -1;
    double used_max_range = 0.0;

    // Retry max range downwards (close-in robustness)
    for (double mr = 4.0; mr >= 2.0 && !ok; mr -= 0.5) {
      used_max_range = mr;
      auto [ok1, th, s, e, tc, dm] =
        followTheGapOnceScored(ranges_eff, i_min, i_max, gap_threshold, mr, theta_goal, alpha);
      if (ok1) { ok = true; theta_final = th; gap_s = s; gap_e = e; theta_c = tc; d_min = dm; }
    }
    if (!ok) {
      for (double mr = 2.0; mr >= 0.5 && !ok; mr -= 0.25) {
        used_max_range = mr;
        auto [ok1, th, s, e, tc, dm] =
          followTheGapOnceScored(ranges_eff, i_min, i_max, gap_threshold, mr, theta_goal, alpha);
        if (ok1) { ok = true; theta_final = th; gap_s = s; gap_e = e; theta_c = tc; d_min = dm; }
      }
    }

    if (ok) last_heading_angle_ = theta_final;
    return {ok, theta_final, gap_s, gap_e, theta_c, d_min, used_max_range};
  }

  // ---------------- Filtering ----------------
  void medianFilterBins(const std::vector<double>& in, std::vector<double>& out, int window)
  {
    window = (window < 3) ? 3 : window;
    if ((window % 2) == 0) window += 1;
    int half = window / 2;

    std::vector<double> tmp;
    tmp.reserve(window);

    for (int i = 0; i < kNumBins; ++i) {
      tmp.clear();
      for (int j = i - half; j <= i + half; ++j) {
        int jj = clampi(j, 0, kNumBins - 1);
        double v = in[jj];
        // Treat inf as very large so it doesn't "inject obstacles"
        tmp.push_back(finite(v) ? v : 1e9);
      }
      std::nth_element(tmp.begin(), tmp.begin() + half, tmp.end());
      double med = tmp[half];
      out[i] = (med >= 1e8) ? inf() : med;
    }
  }

  // Inflate obstacles by blocking neighboring bins based on asin(R_infl / r)
  void inflateObstacles(std::vector<double>& ranges, double r_infl, int max_bins)
  {
    // Copy original (small) to drive inflation without cascading.
    // This is one copy, but fixed size and reused buffer.
    infl_src_ = ranges;

    for (int i = 0; i < kNumBins; ++i) {
      double r = infl_src_[i];
      if (!finite(r)) continue;
      if (r <= 1e-3) continue;

      double ratio = r_infl / r;
      if (ratio <= 0.0) continue;
      if (ratio >= 0.999) ratio = 0.999;

      double dtheta = std::asin(ratio);
      int k = static_cast<int>(std::ceil(dtheta / kAngleRes));
      k = clampi(k, 0, max_bins);

      int lo = clampi(i - k, 0, kNumBins - 1);
      int hi = clampi(i + k, 0, kNumBins - 1);

      for (int j = lo; j <= hi; ++j) {
        // Bring neighbors down toward this obstacle distance (makes them "not free")
        if (!finite(ranges[j])) ranges[j] = r;
        else ranges[j] = std::min(ranges[j], r);
      }
    }
  }

  // ---------------- TF ----------------
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // ---------------- Params ----------------
  std::string base_frame_;
  int forward_sign_;
  int lateral_sign_;
  int cmd_w_sign_;

  double max_speed_;
  double min_speed_;
  double max_yaw_rate_;
  double heading_gain_;
  double gap_threshold_;
  double fov_tight_deg_;
  double fov_wide_deg_;
  double self_radius_;

  double bubble_margin_base_;
  double bubble_margin_kv_;
  int    bubble_max_bins_;

  double score_w_range_;
  double score_w_center_;
  double score_w_edge_;
  int    gap_edge_margin_bins_;

  double near_stop_center_;
  double near_stop_wide_;
  double wall_stop_;

  double ttc_hard_stop_;
  double ttc_soft_start_;
  double ttc_yaw_scale_;

  double accel_up_;
  double accel_down_;

  double yaw_accel_limit_;
  double yaw_min_rate_;
  double yaw_vref_;
  double a_lat_max_;

  double z_min_, z_max_;
  double range_ema_alpha_;
  double range_release_mps_;
  int    median_window_;

  double blend_turn_bias_;
  double blend_straight_bias_;
  double asym_gain_;

  bool   use_curvature_filter_;

  // ---------------- State ----------------
  AdaptivePIDController wall_pid_;

  // If use_curvature_filter_:
  //   filter kappa (curvature), then convert to w = kappa * v.
  // Else:
  //   filter w directly.
  autonomous_driver::Kalman1D steering_filter_{0.02, 0.2};

  pcl::PointCloud<pcl::PointXYZ>::Ptr rgbd_cloud_;
  std::mutex rgbd_mutex_;

  double current_speed_ = 0.0;
  double last_w_cmd_ = 0.0;
  rclcpp::Time last_speed_time_{0, 0, RCL_ROS_TIME};

  double last_heading_angle_ = 0.0;

  // Preallocated bin vectors
  std::vector<double> min_ranges_;
  std::vector<double> ema_ranges_;
  std::vector<double> work_ranges_;
  std::vector<double> median_ranges_;
  std::vector<double> infl_src_;

  // ---------------- Callbacks ----------------
  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // (Left unchanged, still stored but not fused until TF’d)
    if (msg->encoding != "32FC1" && msg->encoding != "16UC1") return;

    double fx = 554.25, fy = 554.25;
    double cx = msg->width / 2.0;
    double cy = msg->height / 2.0;

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->is_dense = false;

    int width  = static_cast<int>(msg->width);
    int height = static_cast<int>(msg->height);

    for (int v = 0; v < height; v += 4) {
      for (int u = 0; u < width; u += 4) {
        int idx = v * width + u;
        float depth = 0.0f;

        if (msg->encoding == "32FC1") {
          const float* p = reinterpret_cast<const float*>(&msg->data[0]);
          depth = p[idx];
        } else {
          const uint16_t* p = reinterpret_cast<const uint16_t*>(&msg->data[0]);
          depth = static_cast<float>(p[idx]) / 1000.0f;
        }

        if (!std::isfinite(depth) || depth <= 0.1f || depth > 20.0f) continue;

        pcl::PointXYZ point;
        point.z = depth;
        point.x = (u - cx) * depth / fx;
        point.y = (v - cy) * depth / fy;
        cloud->points.push_back(point);
      }
    }

    cloud->width  = cloud->points.size();
    cloud->height = 1;

    std::lock_guard<std::mutex> lock(rgbd_mutex_);
    rgbd_cloud_ = cloud;
  }

  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // TF to base_frame_
    sensor_msgs::msg::PointCloud2 cloud_base;
    try {
      cloud_base = tf_buffer_.transform(*msg, base_frame_, tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "TF failed (%s -> %s): %s",
                           msg->header.frame_id.c_str(), base_frame_.c_str(), ex.what());
      return;
    }

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(cloud_base, *cloud);

    const double self_radius_sq = self_radius_ * self_radius_;
    const double max_y = 5.0;
    const double max_range = 50.0;

    // Reset bins (preallocated)
    std::fill(min_ranges_.begin(), min_ranges_.end(), inf());

    double left_min  = max_range;
    double right_min = max_range;

    // corridor metrics
    const double front_cone = 10.0 * M_PI / 180.0;
    const double center_y_max = 0.60;
    const double xf_min_for_front = 0.8;
    double front_far_x = 0.0;
    double front_near_center = inf();
    int front_center_hits = 0;

    for (const auto& pt : cloud->points) {
      // Z-gate: ignore ground/ceiling returns that cause flicker
      if (pt.z < z_min_ || pt.z > z_max_) continue;

      double x = pt.x;
      double y = pt.y;

      // runtime sign normalization
      double xf = static_cast<double>(forward_sign_) * x;
      double yf = static_cast<double>(lateral_sign_) * y;

      double r2 = xf*xf + yf*yf;
      if (r2 < self_radius_sq) continue;
      if (std::abs(yf) > max_y) continue;
      if (xf <= 0.2 || xf > 25.0) continue; // forward window

      double dist = std::hypot(xf, yf);

      // walls
      if (yf > 0.0) left_min = std::min(left_min, dist);
      else         right_min = std::min(right_min, dist);

      double ang = std::atan2(yf, xf);
      if (ang < -kHalfFov || ang > kHalfFov) continue;

      int bin = clampi(static_cast<int>((ang + kHalfFov) / kAngleRes), 0, kNumBins - 1);
      min_ranges_[bin] = std::min(min_ranges_[bin], dist);

      if (std::abs(ang) <= front_cone && std::abs(yf) <= center_y_max && xf >= xf_min_for_front) {
        front_far_x = std::max(front_far_x, xf);
        front_near_center = std::min(front_near_center, xf);
        front_center_hits++;
      }
    }

    if (left_min == max_range)  left_min  = 15.0;
    if (right_min == max_range) right_min = 15.0;
    if (front_center_hits == 0) {
      front_far_x = 12.0;
      front_near_center = 12.0;
    } else if (!std::isfinite(front_near_center)) {
      front_near_center = front_far_x;
    }

    // Wide forward near distance
    const int sector_half_width = 12;
    double front_near_wide = inf();
    int wide_hits = 0;
    for (int i = kForwardBin - sector_half_width; i <= kForwardBin + sector_half_width; ++i) {
      if (i < 0 || i >= kNumBins) continue;
      double r = min_ranges_[i];
      if (!finite(r)) continue;
      front_near_wide = std::min(front_near_wide, r);
      wide_hits++;
    }
    if (wide_hits == 0 || !finite(front_near_wide)) front_near_wide = 50.0;

    // dt for filters/relaxation
    rclcpp::Time now = this->now();
    if (last_speed_time_.nanoseconds() == 0) last_speed_time_ = now;
    double dt = (now - last_speed_time_).seconds();
    dt = clamp(dt, 0.0, 0.1);

    // Temporal smoothing (EMA) + release upwards when bin is missing/inf
    for (int i = 0; i < kNumBins; ++i) {
      double cur = min_ranges_[i];
      double prev = ema_ranges_[i];

      if (finite(cur)) {
        if (!finite(prev)) ema_ranges_[i] = cur;
        else ema_ranges_[i] = lerp(prev, cur, clamp(range_ema_alpha_, 0.01, 1.0));
      } else {
        // If no hit this frame, relax range upward (toward "free")
        if (finite(prev)) {
          double relaxed = prev + range_release_mps_ * dt;
          // Once very large, treat as free
          ema_ranges_[i] = (relaxed > 40.0) ? inf() : relaxed;
        } else {
          ema_ranges_[i] = inf();
        }
      }
    }

    // Median filter across bins to reduce flicker
    medianFilterBins(ema_ranges_, median_ranges_, median_window_);

    // Adaptive FOV (wide on open straights, tight near walls/clutter)
    // Use front_far_x and proximity to tighten.
    const double fov_tight = clamp(fov_tight_deg_, 5.0, 90.0) * M_PI / 180.0;
    const double fov_wide  = clamp(fov_wide_deg_,  5.0, 90.0) * M_PI / 180.0;

    // Map free space ahead -> [0..1]
    const double fov_far_stop = 4.0;
    const double fov_far_free = 18.0;
    double t_open = clamp((front_far_x - fov_far_stop) / (fov_far_free - fov_far_stop), 0.0, 1.0);

    double closest_wall = std::min(left_min, right_min);
    double wall_tighten = clamp((1.2 - closest_wall) / 1.2, 0.0, 1.0); // closer wall => tighten
    double t = clamp(t_open * (1.0 - 0.85 * wall_tighten), 0.0, 1.0);

    double used_fov = lerp(fov_tight, fov_wide, t);
    used_fov = clamp(used_fov, 5.0 * M_PI/180.0, 90.0 * M_PI/180.0);

    int i_min = clampi(angleToBin(-used_fov), 0, kNumBins - 1);
    int i_max = clampi(angleToBin(+used_fov), 0, kNumBins - 1);
    if (i_min > i_max) std::swap(i_min, i_max);

    // Build effective ranges (masked outside FOV and capped)
    // (No copies per FTG; we reuse work_ranges_)
    for (int i = 0; i < kNumBins; ++i) {
      double a = binToAngle(i);
      if (std::abs(a) > used_fov) {
        work_ranges_[i] = inf();
        continue;
      }
      double r = median_ranges_[i];
      if (!finite(r) || r > 25.0) work_ranges_[i] = inf();
      else work_ranges_[i] = r;
    }

    // Speed-aware inflation radius
    double bubble_margin = bubble_margin_base_ + bubble_margin_kv_ * std::max(0.0, current_speed_);
    double r_infl = std::max(0.0, self_radius_ + bubble_margin);
    inflateObstacles(work_ranges_, r_infl, bubble_max_bins_);

    // Wall-follow
    double wall_error = left_min - right_min;
    double wall_yaw = wall_pid_.compute(wall_error, closest_wall, now);

    // FTG (scored)
    auto [ftg_ok, theta_final, gap_s, gap_e, theta_c, d_min, used_max_range] =
      followTheGapAdaptiveScored(work_ranges_, i_min, i_max, gap_threshold_);

    // fallback
    if (!ftg_ok && finite(last_heading_angle_) && std::abs(last_heading_angle_) > 1e-6) {
      theta_final = last_heading_angle_;
      ftg_ok = true;
    }

    // Convert heading -> yaw
    const double max_heading_angle = M_PI / 3.0;
    double theta_cmd = clamp(theta_final, -max_heading_angle, max_heading_angle);
    double ftg_yaw = heading_gain_ * theta_cmd;

    // Improved blending: based on turn demand + corridor asymmetry + proximity
    double abs_turn = clamp(std::abs(theta_cmd) / max_heading_angle, 0.0, 1.0);
    double base_w_ftg = lerp(blend_straight_bias_, blend_turn_bias_, abs_turn);

    double asym = 0.0;
    {
      // corridor asymmetry: if one side is much closer, trust FTG more
      double denom = std::max(0.5, left_min + right_min);
      asym = (left_min - right_min) / denom; // [-1..1]
    }
    base_w_ftg = clamp(base_w_ftg + asym_gain_ * std::abs(asym), 0.20, 0.95);

    // proximity: if very close to walls, FTG should dominate (avoid wall PID fighting)
    double prox = clamp((1.0 - closest_wall) / 1.0, 0.0, 1.0);
    base_w_ftg = clamp(base_w_ftg + 0.35 * prox, 0.20, 0.97);

    double steering = 0.0;
    if (!ftg_ok) steering = wall_yaw;
    else steering = base_w_ftg * ftg_yaw + (1.0 - base_w_ftg) * wall_yaw;

    // --- TTC-based braking and yaw clamp ---
    double v_now = std::max(current_speed_, 0.1);
    double ttc = front_near_center / v_now;

    // distance hard-stop guard still applies
    bool dist_hard_stop = (front_near_center <= near_stop_center_) ||
                          (front_near_wide   <= near_stop_wide_)   ||
                          (closest_wall      <= wall_stop_);

    bool ttc_hard_stop = (ttc <= ttc_hard_stop_);

    // Soft scaling region
    double soft_scale = 1.0;
    if (ttc < ttc_soft_start_) {
      soft_scale = clamp((ttc - ttc_hard_stop_) / std::max(1e-3, (ttc_soft_start_ - ttc_hard_stop_)), 0.0, 1.0);
    }

    // Speed-dependent yaw clamp (tighten at higher speeds)
    // max_yaw_dyn goes from max_yaw_rate_ at low speed toward yaw_min_rate_ at high speed
    double vref = std::max(0.5, yaw_vref_);
    double yaw_t = clamp(vref / std::max(v_now, vref), 0.0, 1.0); // <=1; decreases as v increases
    double max_yaw_dyn = clamp(lerp(yaw_min_rate_, max_yaw_rate_, yaw_t), yaw_min_rate_, max_yaw_rate_);

    // Apply TTC yaw scaling near obstacles
    if (ttc < ttc_soft_start_) {
      double yaw_scale = lerp(ttc_yaw_scale_, 1.0, soft_scale); // smaller when TTC small
      max_yaw_dyn *= yaw_scale;
    }

    steering = clamp(steering, -max_yaw_dyn, +max_yaw_dyn);

    // --- Speed planning ---
    double v_des = 0.0;
    double w_cmd = steering;

    // Lat-accel based curve speed (using yaw rate as proxy; w = v/R -> v <= a_lat / |w|)
    // This is conservative; still very effective.
    const double w_eps = 0.05;
    double v_curve = max_speed_;
    double abs_w = std::abs(w_cmd);
    if (abs_w > w_eps) v_curve = std::min(max_speed_, a_lat_max_ / abs_w);

    // Far distance-based speed (your original idea)
    const double far_stop = 2.0;
    const double far_free = 22.0;
    double t_far = clamp((front_far_x - far_stop) / (far_free - far_stop), 0.0, 1.0);
    double v_far = lerp(min_speed_, max_speed_, t_far);

    v_des = std::min(v_far, v_curve);

    // Apply TTC soft scale to speed
    if (ttc < ttc_soft_start_) {
      // keep some minimum rolling unless extremely close
      double v_min_roll = 0.35;
      v_des = lerp(v_min_roll, v_des, soft_scale);
    }

    // Hard stops: do NOT spin in place into a wall
    if (dist_hard_stop || ttc_hard_stop) {
      v_des = 0.0;
      w_cmd = 0.0;
    }

    // Curvature filtering recommended:
    //   kappa = w/v, filter kappa, then w = kappa*v
    // else filter w directly
    if (use_curvature_filter_) {
      double v_for_kappa = std::max(v_des, 0.25);
      double kappa_des = w_cmd / v_for_kappa;

      steering_filter_.predict();
      steering_filter_.update(kappa_des);
      double kappa = steering_filter_.value();

      w_cmd = kappa * std::max(current_speed_, 0.25);
    } else {
      steering_filter_.predict();
      steering_filter_.update(w_cmd);
      w_cmd = steering_filter_.value();
    }

    // Apply runtime sign for cmd_vel consumer
    w_cmd *= static_cast<double>(cmd_w_sign_);

    // Yaw slew limiter (prevents snap at high Hz)
    double w_max_delta = yaw_accel_limit_ * dt;
    w_cmd = approach(last_w_cmd_, w_cmd, w_max_delta);
    last_w_cmd_ = w_cmd;

    // Final publish with speed slew and throttled logs
    applySpeedSlewAndPublish(v_des, w_cmd,
                             left_min, right_min, wall_error, closest_wall,
                             ftg_ok, theta_final, theta_c, d_min, gap_s, gap_e,
                             used_max_range, used_fov,
                             front_near_center, front_far_x, front_near_wide,
                             front_center_hits, ttc, base_w_ftg);
  }

  void applySpeedSlewAndPublish(double v_des, double w_cmd,
                                double left_min, double right_min, double wall_error,
                                double closest_wall,
                                bool ftg_ok, double theta_final, double theta_c, double d_min,
                                int gap_s, int gap_e,
                                double used_max_range, double used_fov,
                                double front_near_center, double front_far_x, double front_near_wide, int hits,
                                double ttc, double w_ftg)
  {
    rclcpp::Time now = this->now();
    if (last_speed_time_.nanoseconds() == 0) last_speed_time_ = now;
    double dt = (now - last_speed_time_).seconds();
    last_speed_time_ = now;
    dt = clamp(dt, 0.0, 0.1);

    double max_up = accel_up_ * dt;
    double max_dn = accel_down_ * dt;

    if (v_des > current_speed_) current_speed_ = approach(current_speed_, v_des, max_up);
    else                        current_speed_ = approach(current_speed_, v_des, max_dn);

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x  = current_speed_;
    cmd.angular.z = w_cmd;

    auto deg = [](double r){ return r * 180.0 / M_PI; };

    // Throttle logs for performance
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 250,
      "L=%.2f R=%.2f err=%.2f closest=%.2f | front_near=%.2f wide=%.2f far=%.2f hits=%d ttc=%.2f | "
      "v_des=%.2f v=%.2f | w=%.2f | blend_ftg=%.2f | FTG[%s] theta=%.1f° tc=%.1f° dmin=%.2f gap=[%d..%d] fov=%.1f° maxR=%.2f | "
      "SIGNS fwd=%d lat=%d cmdw=%d",
      left_min, right_min, wall_error, closest_wall,
      front_near_center, front_near_wide, front_far_x, hits, ttc,
      v_des, current_speed_, w_cmd, w_ftg,
      (ftg_ok ? "OK" : "FAIL"),
      deg(theta_final), deg(theta_c), d_min, gap_s, gap_e, deg(used_fov), used_max_range,
      forward_sign_, lateral_sign_, cmd_w_sign_);

    cmd_pub_->publish(cmd);
    publishSteeringArc(cmd.angular.z, cmd.linear.x);
  }

  void publishSteeringArc(double yaw_rate, double speed)
  {
    visualization_msgs::msg::Marker arc;
    arc.header.frame_id = base_frame_;
    arc.header.stamp = this->now();
    arc.ns = "steering";
    arc.id = 0;
    arc.type = visualization_msgs::msg::Marker::LINE_STRIP;
    arc.action = visualization_msgs::msg::Marker::ADD;
    arc.pose.orientation.w = 1.0;
    arc.scale.x = 0.05;
    arc.color.a = 1.0;
    arc.color.r = 1.0;
    arc.color.g = 1.0;
    arc.color.b = 0.0;
    arc.lifetime = rclcpp::Duration::from_seconds(0.1);

    const double dt = 0.1;
    const double total_time = 1.5;
    int steps = static_cast<int>(total_time / dt);

    double x = 0.0, y = 0.0, theta = 0.0;
    for (int i = 0; i < steps; ++i) {
      geometry_msgs::msg::Point p;
      p.x = x; p.y = y; p.z = 0.1;
      arc.points.push_back(p);

      double v = speed;
      double w = yaw_rate;
      x += v * dt * std::cos(theta);
      y += v * dt * std::sin(theta);
      theta += w * dt;
    }

    marker_pub_->publish(arc);
  }

  // ---------------- ROS interfaces ----------------
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FusionDriver>());
  rclcpp::shutdown();
  return 0;
}
