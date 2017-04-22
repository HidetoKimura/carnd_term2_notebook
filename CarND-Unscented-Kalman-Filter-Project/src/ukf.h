#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  //VectorXd x_;

  ///* state covariance matrix
  // MatrixXd P_;

  ///* predicted sigma points matrix
  //MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  //VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  //double lambda_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  ///* measurement dimension, radar can measure r, phi, and r_dot
  int n_z_radar_;
  int n_z_laser_;

  //define spreading parameter
  double lambda_;

  //time diff in sec
  double delta_t_; 
  
  //set state
  VectorXd x_;

  //set covariance matrix
  MatrixXd P_;

  //create sigma point matrix
  MatrixXd Xsig_aug_;

   //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred_;

  //create vector for weights
  VectorXd weights_;
  
  //mean predicted measurement
  VectorXd z_pred_radar_;

  //measurement covariance matrix S
  MatrixXd S_radar_;
 
  //create example matrix with sigma points in measurement space
  MatrixXd Zsig_radar_;

  //create example vector for incoming radar measurement
  VectorXd z_radar_;

  //mean predicted measurement
  VectorXd z_pred_laser_;

  //measurement covariance matrix S
  MatrixXd S_laser_;
 
  //create example matrix with sigma points in measurement space
  MatrixXd Zsig_laser_;

  //create example vector for incoming radar measurement
  VectorXd z_laser_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
  /**
   * Ukf functions
   */
  void AugmentedSigmaPoints(void);
  void SigmaPointPrediction(void);
  void PredictMeanAndCovariance(void);
  void PredictRadarMeasurement(void);
  void PredictLaserMeasurement(void);
  void UpdateStateRadar(void);
  void UpdateStateLaser(void);
};

#endif /* UKF_H */
