// Controlled, repeatable force disturbance for Gazebo Classic + ROS 2 Humble.
//
// The mouse only moves the visual marker. On /disturbance/apply, this plugin:
//   1. Reads the marker's world position.
//   2. Converts it to a point fixed in the target link frame.
//   3. Applies a half-sine force pulse at that point every physics update.
//
// This avoids collision variability and avoids competing with ros2_control
// command publishers.

#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/UpdateInfo.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace wheel_leg_description
{

constexpr double kPi = 3.14159265358979323846;

class ControlledDisturbancePlugin : public gazebo::ModelPlugin
{
public:
  ControlledDisturbancePlugin() = default;
  ~ControlledDisturbancePlugin() override = default;

  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = std::move(model);
    world_ = model_->GetWorld();
    ros_node_ = gazebo_ros::Node::Get(sdf);

    target_link_name_ = GetSdf<std::string>(sdf, "target_link", "base_link");
    marker_model_name_ =
      GetSdf<std::string>(sdf, "marker_model", "disturbance_marker");
    force_frame_ = GetSdf<std::string>(sdf, "force_frame", "world");
    peak_force_ = GetSdf<double>(sdf, "peak_force", 10.0);
    duration_ = GetSdf<double>(sdf, "duration", 0.10);
    direction_ = GetSdf<ignition::math::Vector3d>(
      sdf, "force_direction", ignition::math::Vector3d(1.0, 0.0, 0.0));

    if (peak_force_ <= 0.0) {
      RCLCPP_FATAL(ros_node_->get_logger(), "peak_force must be > 0.");
      return;
    }
    if (duration_ <= 0.0) {
      RCLCPP_FATAL(ros_node_->get_logger(), "duration must be > 0.");
      return;
    }
    if (direction_.Length() < 1e-9) {
      RCLCPP_FATAL(ros_node_->get_logger(), "force_direction must be non-zero.");
      return;
    }
    direction_.Normalize();

    if (force_frame_ != "world" && force_frame_ != "link") {
      RCLCPP_FATAL(
        ros_node_->get_logger(),
        "force_frame must be either 'world' or 'link'.");
      return;
    }

    target_link_ = model_->GetLink(target_link_name_);
    if (!target_link_) {
      RCLCPP_FATAL(
        ros_node_->get_logger(),
        "Target link '%s' was not found in model '%s'.",
        target_link_name_.c_str(), model_->GetName().c_str());
      return;
    }

    apply_service_ = ros_node_->create_service<std_srvs::srv::Trigger>(
      "apply",
      std::bind(
        &ControlledDisturbancePlugin::OnApplyRequest,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    cancel_service_ = ros_node_->create_service<std_srvs::srv::Trigger>(
      "cancel",
      std::bind(
        &ControlledDisturbancePlugin::OnCancelRequest,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(
        &ControlledDisturbancePlugin::OnUpdate,
        this,
        std::placeholders::_1));

    const double nominal_impulse =
      2.0 * peak_force_ * duration_ / kPi;

    RCLCPP_INFO(
      ros_node_->get_logger(),
      "Controlled disturbance ready: target_link=%s marker_model=%s "
      "peak_force=%.3f N duration=%.4f s impulse=%.4f N*s "
      "direction=[%.3f %.3f %.3f] frame=%s",
      target_link_name_.c_str(),
      marker_model_name_.c_str(),
      peak_force_,
      duration_,
      nominal_impulse,
      direction_.X(),
      direction_.Y(),
      direction_.Z(),
      force_frame_.c_str());
  }

private:
  template<typename T>
  T GetSdf(
    const sdf::ElementPtr & sdf,
    const std::string & name,
    const T & default_value)
  {
    if (sdf && sdf->HasElement(name)) {
      return sdf->Get<T>(name);
    }
    return default_value;
  }

  void OnApplyRequest(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    trigger_requested_.store(true);
    cancel_requested_.store(false);
    response->success = true;
    response->message =
      "Disturbance queued for the next Gazebo physics update.";
  }

  void OnCancelRequest(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    cancel_requested_.store(true);
    trigger_requested_.store(false);
    response->success = true;
    response->message = "Disturbance cancellation queued.";
  }

  void OnUpdate(const gazebo::common::UpdateInfo & info)
  {
    if (cancel_requested_.exchange(false)) {
      active_ = false;
      RCLCPP_INFO(ros_node_->get_logger(), "Disturbance cancelled.");
    }

    if (trigger_requested_.exchange(false)) {
      StartPulse(info.simTime);
    }

    if (!active_) {
      return;
    }

    const double elapsed = (info.simTime - start_time_).Double();
    if (elapsed < 0.0) {
      return;
    }

    if (elapsed >= duration_) {
      active_ = false;
      RCLCPP_INFO(
        ros_node_->get_logger(),
        "Disturbance completed at local point [%.4f %.4f %.4f] m.",
        application_point_local_.X(),
        application_point_local_.Y(),
        application_point_local_.Z());
      return;
    }

    const double phase = kPi * elapsed / duration_;
    const double amplitude = peak_force_ * std::sin(phase);
    const ignition::math::Pose3d link_pose = target_link_->WorldPose();

    ignition::math::Vector3d direction_world;
    if (force_frame_ == "link") {
      direction_world = link_pose.Rot().RotateVector(direction_);
    } else {
      direction_world = direction_;
    }

    const ignition::math::Vector3d force_world = amplitude * direction_world;
    const ignition::math::Vector3d point_world =
      link_pose.Pos() + link_pose.Rot().RotateVector(application_point_local_);

    target_link_->AddForceAtWorldPosition(force_world, point_world);
  }

  void StartPulse(const gazebo::common::Time & sim_time)
  {
    const gazebo::physics::ModelPtr marker =
      world_->ModelByName(marker_model_name_);

    if (!marker) {
      RCLCPP_ERROR(
        ros_node_->get_logger(),
        "Marker model '%s' was not found. Spawn it before applying a pulse.",
        marker_model_name_.c_str());
      active_ = false;
      return;
    }

    const ignition::math::Pose3d marker_pose = marker->WorldPose();
    const ignition::math::Pose3d link_pose = target_link_->WorldPose();

    application_point_local_ =
      link_pose.Rot().RotateVectorReverse(
        marker_pose.Pos() - link_pose.Pos());

    start_time_ = sim_time;
    active_ = true;

    const double nominal_impulse =
      2.0 * peak_force_ * duration_ / kPi;

    RCLCPP_INFO(
      ros_node_->get_logger(),
      "Disturbance started: marker_world=[%.4f %.4f %.4f] m, "
      "point_local=[%.4f %.4f %.4f] m, impulse=%.4f N*s.",
      marker_pose.Pos().X(),
      marker_pose.Pos().Y(),
      marker_pose.Pos().Z(),
      application_point_local_.X(),
      application_point_local_.Y(),
      application_point_local_.Z(),
      nominal_impulse);
  }

  gazebo::physics::ModelPtr model_;
  gazebo::physics::WorldPtr world_;
  gazebo::physics::LinkPtr target_link_;
  gazebo::event::ConnectionPtr update_connection_;

  gazebo_ros::Node::SharedPtr ros_node_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr apply_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cancel_service_;

  std::string target_link_name_;
  std::string marker_model_name_;
  std::string force_frame_;

  ignition::math::Vector3d direction_{1.0, 0.0, 0.0};
  ignition::math::Vector3d application_point_local_{0.0, 0.0, 0.0};

  double peak_force_{10.0};
  double duration_{0.10};

  gazebo::common::Time start_time_;
  bool active_{false};

  std::atomic<bool> trigger_requested_{false};
  std::atomic<bool> cancel_requested_{false};
};

GZ_REGISTER_MODEL_PLUGIN(ControlledDisturbancePlugin)

}  // namespace wheel_leg_description
