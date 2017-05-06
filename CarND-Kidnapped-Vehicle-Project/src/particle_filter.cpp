/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <iomanip>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// noise generation
	random_device seed_gen;
	default_random_engine gen(seed_gen());
	
	normal_distribution<double> ndist_x(x, std[0]);
	normal_distribution<double> ndist_y(y, std[1]);
	normal_distribution<double> ndist_theta(theta, std[2]);
	
	num_particles = 1000;

	weights.resize(num_particles);
	particles.resize(num_particles);	
	
	for (int i = 0; i < num_particles; i++) {
		particles[i].id = i;
		particles[i].x = ndist_x(gen);
		particles[i].y = ndist_y(gen);
		particles[i].theta = ndist_theta(gen);
		particles[i].weight = 1.0;
		weights[i] = 1.0;
	}
	//write("./data/output/init_particles.txt");

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	random_device seed_gen;
	default_random_engine gen(seed_gen());
	
	normal_distribution<double> ndist_x(0, std_pos[0]);
	normal_distribution<double> ndist_y(0, std_pos[1]);
	normal_distribution<double> ndist_theta(0, std_pos[2]);

	double yaw  = delta_t * yaw_rate;

	for (int i = 0; i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		/* prediction */
		if(fabs(yaw_rate) > 0.001) {
			x += velocity / yaw_rate * (sin(theta + yaw) - sin(theta));
			y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw));
			theta += yaw;
		}
		else {
			x += velocity * delta_t * cos(theta);
			y += velocity * delta_t * sin(theta);
		}
		/* add noise */
		particles[i].x = x + ndist_x(gen);
		particles[i].y = y + ndist_y(gen); 
		particles[i].theta = theta + ndist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observasions) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(auto& obs: observasions) {
		double shortest = 10000.0;
		int shortest_id = -1;
		for(auto& map: predicted) {
			double distance = dist(map.x, map.y, obs.x, obs.y);
			if(shortest > distance) {
				shortest = distance;
				shortest_id = map.id;	
			}
		}
		obs.id = shortest_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observasions, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observasions are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i = 0; i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
	
		int num_obs = observasions.size();
		
		vector<LandmarkObs> global_obs;
		global_obs.resize(num_obs);

		/* translate MAP's coodinate system */
		for(int j = 0; j < num_obs; j++ ) {
			double v_x = observasions[j].x;
			double v_y = observasions[j].y;
			global_obs[j].x = x + v_x * cos(theta) - v_y * sin(theta);
			global_obs[j].y = y + v_x * sin(theta) + v_y * cos(theta); 
			global_obs[j].id = -1;
		}

		/* choose landmarks within sensor_range */
	    vector<LandmarkObs> predicted;
    	for (auto& landmark : map_landmarks.landmark_list) {
      		if (dist(x, y, landmark.x_f, landmark.y_f) < sensor_range) {
				LandmarkObs lm_obj;
        		lm_obj.id = landmark.id_i;	
				lm_obj.x  = landmark.x_f;
				lm_obj.y  = landmark.y_f;
        		predicted.push_back(lm_obj);
      		}
		}

		dataAssociation(predicted, global_obs);

		/* calculate weight */
		double all_weight = 1.0;
		for(int j = 0; j < num_obs; j++ ) {
			int index = global_obs[j].id;
			if(index == -1) continue;
			double mu_x = map_landmarks.landmark_list[index - 1].x_f;
			double mu_y = map_landmarks.landmark_list[index - 1].y_f;
			double x = global_obs[j].x;
			double y = global_obs[j].y;
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1]; 
			double c1 = 1.0 / ( 2.0 * M_PI * sig_x * sig_y);
			double c2 = pow(x - mu_x, 2) / (2.0 * sig_x * sig_x);
			double c3 = pow(y - mu_y, 2) / (2.0 * sig_y * sig_y);
			double weight = c1 * exp( -1.0 * (c2 * c3));
			//cout << "particle = " << i << ", obj = " << j << ", index = " << index << ", weight = " << weight << endl;
			all_weight *= weight;
		}
		//cout << "particle = " << i << ", all_weight = " << all_weight << endl;
		particles[i].weight = all_weight;
		weights[i] = all_weight;
    }
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device seed_gen;
	default_random_engine gen(seed_gen());
	
	discrete_distribution<int> ddist_index(weights.begin(), weights.end());

	vector<Particle> resample_particles;
	resample_particles.resize(num_particles);
	for(int i = 0; i < num_particles; i++) {
		resample_particles[i] = particles[ddist_index(gen)];
	}

	particles = resample_particles;
	//static int count = 0;
	//ostringstream ostr;
	//ostr << "./data/output/resampled_" << setfill('0') << setw(6) << count++ << ".txt";
	//write(ostr.str());
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
