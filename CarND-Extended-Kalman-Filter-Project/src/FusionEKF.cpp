#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {

  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  x_ = VectorXd(4);
  P_ = MatrixXd(4, 4);
  F_ = MatrixXd(4, 4);
  Q_ = MatrixXd(4, 4);
 
  //measurement covariance matrix - laser
/*
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
*/
  R_radar_ << 0.014412589090776581, 0, 0,
    0, 1.3610836622321855e-06, 0,
    0, 0, 0.011073356944289297;
  R_laser_ << 0.0068374897772981421, 0,
    0, 0.0054887300686829819;


  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  //state covariance matrix P
/*
  P_ << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1000, 0,
    0, 0, 0, 1000;
*/
  P_ << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1;

  //measurement matrix
  H_laser_ << 1, 0, 0, 0,
    0, 1, 0, 0;

  //the initial transition matrix F_
  F_ << 1, 0, 1, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,
    0, 0, 0, 1;


  //set the acceleration noise components
/*
  noise_ax = 9;
  noise_ay = 9;
 */
  noise_ax = 4.5;
  noise_ay = 4.5;
 
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      x_ << ro * cos(phi), ro * sin(phi), 0, 0;
      ekf_.Init(x_, P_, F_, Hj_, R_radar_, Q_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
      ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
    }

    // update timestamp
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

 
  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  F_(0, 2) = dt;
  F_(1, 3) = dt;

  //set the process covariance matrix Q
  Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
     0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
     dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
     0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
   if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // set radar parameters
    ekf_.Init(x_, P_, F_, Hj_, R_radar_, Q_);
  } else {
    // set laser prameters s
    ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
  }
  
  ekf_.Predict();
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Calculate Jacobian  
    Hj_ = tools.CalculateJacobian(x_);
    ekf_.Init(ekf_.x_, ekf_.P_, F_, Hj_, R_radar_, Q_);
    // Radar updates
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
