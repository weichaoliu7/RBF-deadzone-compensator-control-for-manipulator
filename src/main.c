#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include "sine.h"
#include "cosine.h"
#include "inv_matrix.h"

// reference: [1]Liu JinKun. Robot Control System Design and MATLAB Simulation[M]. Tsinghua University Press, 2008.
// [2]Lewis F L, Campos J, Selmic R. Neuro-fuzzy control of industrial systems with actuator nonlinearities[M]. Society for Industrial and Applied Mathematics, 2002.
// [3]Lu Y, Liu J K, Sun F C. Actuator nonlinearities compensation using RBF neural networks in robot control system[C]//The Proceedings of the Multiconference on" Computational Engineering in Systems Applications". IEEE, 2006, 1: 231-238.

// global variables declaration
#define PI 3.14159
#define H 7              // hidden layer neurons number
#define n_joint 2        // output layer neurons number
#define ARRAY_SIZE 20000 // sampling times

static double Ts = 0.001; // sampling period
static double t0 = 0.0;   // start time
static double t1 = 20.0;  // end time
static double center_compensator[n_joint][H] = {{-40, -25, -10, 0, 10, 25, 40},
                                                {-40, -25, -10, 0, 10, 25, 40}}; // compensator neural network's center
static double center_controller[n_joint][H] = {{-40, -25, -10, 0, 10, 25, 40},
                                               {-40, -25, -10, 0, 10, 25, 40}}; // controller neural network's center
static double width_compensator = 100;                                          // compensator neural network's width
static double width_controller = 100;                                           // controller neural network's width
static double p[] = {2.9, 0.76, 0.87, 3.04, 0.87};
static double g = 9.8; // gravitational acceleration

double frobenius_norm(int m, double A[m]){
    double norm = 0.0;
    for (int h = 0; h < m; h++){
        norm += A[h] * A[h];
    }
    return sqrt(norm);
}

double torque[n_joint]; // control input torque

struct _archive{
    double q1_archive[ARRAY_SIZE];
    double dq1_archive[ARRAY_SIZE];
    double q2_archive[ARRAY_SIZE];
    double dq2_archive[ARRAY_SIZE];
    double error1_archive[ARRAY_SIZE];
    double error2_archive[ARRAY_SIZE];
    double error1_velocity_archive[ARRAY_SIZE];
    double error2_velocity_archive[ARRAY_SIZE];
    double controller_output1_archive[ARRAY_SIZE];
    double controller_output2_archive[ARRAY_SIZE];
    double compensator_output1_archive[ARRAY_SIZE];
    double compensator_output2_archive[ARRAY_SIZE];
    double torque1_archive[ARRAY_SIZE];
    double torque2_archive[ARRAY_SIZE];
    double sat_function1_archive[ARRAY_SIZE];
    double sat_function2_archive[ARRAY_SIZE];
} archive;

Data q1_desired, dq1_desired, ddq1_desired;
Data q2_desired, dq2_desired, ddq2_desired;

struct Amp{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

struct M0{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

struct B0{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

void SystemInput(Data *q1_desired, Data *dq1_desired, Data *ddq1_desired, Data *q2_desired, Data *dq2_desired, Data *ddq2_desired, double Ts, double t0, double t1){

    struct Amp amp; // amplitude
    amp.q1_desired = 2;
    amp.dq1_desired = 2 * 0.2 * PI;
    amp.ddq1_desired = -2 * pow(0.2 * PI, 2);
    amp.q2_desired = 1;
    amp.dq2_desired = -0.2 * PI;
    amp.ddq2_desired = -pow(0.2 * PI, 2);

    struct M0 m0; // angular frequency
    m0.q1_desired = 0.2 * PI;
    m0.dq1_desired = 0.2 * PI;
    m0.ddq1_desired = 0.2 * PI;
    m0.q2_desired = 0.2 * PI;
    m0.dq2_desired = 0.2 * PI;
    m0.ddq2_desired = 0.2 * PI;

    struct B0 b0; // vertical shift
    b0.q1_desired = 0;
    b0.dq1_desired = 0;
    b0.ddq1_desired = 0;
    b0.q2_desired = 0;
    b0.dq2_desired = 0;
    b0.ddq2_desired = 0;

    sine(q1_desired, Ts, t0, t1, amp.q1_desired, m0.q1_desired, b0.q1_desired);           // desired angular displacement of link 1
    cosine(dq1_desired, Ts, t0, t1, amp.dq1_desired, m0.dq1_desired, b0.dq1_desired);     // desired angular velocity of link 1
    sine(ddq1_desired, Ts, t0, t1, amp.ddq1_desired, m0.ddq1_desired, b0.ddq1_desired);   // desired angular acceleration of link 1
    cosine(q2_desired, Ts, t0, t1, amp.q2_desired, m0.q2_desired, b0.q2_desired);         // desired angular displacement of link 2
    sine(dq2_desired, Ts, t0, t1, amp.dq2_desired, m0.dq2_desired, b0.dq2_desired);       // desired angular velocity of link 2
    cosine(ddq2_desired, Ts, t0, t1, amp.ddq2_desired, m0.ddq2_desired, b0.ddq2_desired); // desired angular acceleration of link 2
}

struct _system_state{
    double q1;   // actual angular displacement of link 1
    double dq1;  // actual angular velocity of link 1
    double ddq1; // actual angular acceleration of link 1
    double q2;   // actual angular displacement of link 2
    double dq2;  // actual angular velocity of link 2
    double ddq2; // actual angular acceleration of link 2
} system_state;

// neural network structure, first neural network is related to compensator
struct _compensator_NN_structure{
    double input[n_joint];                       // compensator neural network's input
    double neural_output[n_joint];               // compensator neural network's output
    double hidden_output[n_joint][H];            // compensator neural network's hidden layer output
    double hidden_output_derivative[n_joint][H]; // compensator neural network's hidden layer output's derivative
    double weight[n_joint][H];                   // compensator neural network's weight estimate
    double weight_derivative[n_joint][H];        // compensator neural network's weight estimate's derivative
} compensator_NN_structure;

// neural network structure, second neural network is related to controller
struct _controller_NN_structure{
    double input[n_joint];                       // controller neural network's input
    double neural_output[n_joint];               // controller neural network's output
    double hidden_output[n_joint][H];            // controller neural network's hidden layer output
    double hidden_output_derivative[n_joint][H]; // controller neural network's hidden layer output's derivative
    double weight[n_joint][H];                   // controller neural network's weight estimate
    double weight_derivative[n_joint][H];        // controller neural network's weight estimate's derivative
} controller_NN_structure;

struct _deadzone{
    double start_deadzone; // start of dead zone
    double end_deadzone;   // end of dead zone
} deadzone;

struct _dynamics{
    double M[n_joint][n_joint];  // manipulator's inertia matrix
    double G[n_joint];           // manipulator's gravity matrix
    double Vm[n_joint][n_joint]; // manipulator's Coriolis matrix
} dynamics;

struct _controller{
    double controller_u[10];
    double controller_out[4];
    double controller_output[n_joint]; // controller output
    double error[n_joint];             // angular displacement error
    double error_velocity[n_joint];    // angular velocity error
    double Lambda[n_joint][n_joint];   // defined in Eq. 34
    double r[n_joint];                 // filtered tracking error, defined in Eq. 34
    double dqr[n_joint];               // derivative of r
    double ddqr[n_joint];              // second-order derivative of r
    double Kv[n_joint];                // feedback gain, defined in Eq. 40
} controller;

void CONTROLLER_init(){
    system_state.q1 = 0.2;
    system_state.dq1 = 0.0;
    system_state.q2 = 0.2;
    system_state.dq2 = 0.0;
    controller.controller_u[0] = q1_desired.y[0];
    controller.controller_u[1] = dq1_desired.y[0];
    controller.controller_u[2] = ddq1_desired.y[0];
    controller.controller_u[3] = q2_desired.y[0];
    controller.controller_u[4] = dq2_desired.y[0];
    controller.controller_u[5] = ddq2_desired.y[0];
    controller.controller_u[6] = system_state.q1;
    controller.controller_u[7] = system_state.dq1;
    controller.controller_u[8] = system_state.q2;
    controller.controller_u[9] = system_state.dq2;
    controller.Kv[0] = 20;
    controller.Kv[1] = 20;
}

struct _compensator{
    double compensator_u[4];
    double compensator_out[2];
    double compensator_output[n_joint]; // neural network compensator output, denoted u
    double S;
    double T;
    double k1;
    double k2;
    double sat_function[n_joint]; // asymmetric saturation function
} compensator;

void COMPENSATOR_init(){
    controller_NN_structure.neural_output[0] = 0.0;
    controller_NN_structure.neural_output[1] = 0.0;
    deadzone.start_deadzone = -10;
    deadzone.end_deadzone = 10;
    compensator.S = 600;
    compensator.T = 600;
    compensator.k1 = 0.0001;
    compensator.k2 = 0.0001;

    for (int j = 0; j < n_joint; j++) {
        controller.Lambda[j][j] = 5.0;
    }

    for (int j = 0; j < H; j++){
        compensator_NN_structure.weight[0][j] = 20.0;
        compensator_NN_structure.weight[1][j] = 20.0;
        controller_NN_structure.weight[0][j] = 20.0;
        controller_NN_structure.weight[1][j] = 20.0;
    }
}

double CONTROLLER_realize(int i){
    controller.controller_u[0] = q1_desired.y[i];
    controller.controller_u[1] = dq1_desired.y[i];
    controller.controller_u[2] = ddq1_desired.y[i];
    controller.controller_u[3] = q2_desired.y[i];
    controller.controller_u[4] = dq2_desired.y[i];
    controller.controller_u[5] = ddq2_desired.y[i];
    controller.controller_u[6] = system_state.q1;
    controller.controller_u[7] = system_state.dq1;
    controller.controller_u[8] = system_state.q2;
    controller.controller_u[9] = system_state.dq2;
    // printf("dq1_desired.y[%d] = %f\n", i, dq1_desired.y[i]);

    archive.q1_archive[i] = controller.controller_u[6];
    archive.dq1_archive[i] = controller.controller_u[7];
    archive.q2_archive[i] = controller.controller_u[8];
    archive.dq2_archive[i] = controller.controller_u[9];

    controller.error[0] = q1_desired.y[i] - system_state.q1;            // angular position tracking error of link 1
    controller.error_velocity[0] = dq1_desired.y[i] - system_state.dq1; // angular velocity tracking error of link 1
    controller.error[1] = q2_desired.y[i] - system_state.q2;            // angular position tracking error of link 2
    controller.error_velocity[1] = dq2_desired.y[i] - system_state.dq2; // angular velocity tracking error of link 2
    // printf("controller.error_velocity[1] = %f\n", controller.error_velocity[1]);

    archive.error1_archive[i] = controller.error[0];
    archive.error1_velocity_archive[i] = controller.error_velocity[0];
    archive.error2_archive[i] = controller.error[1];
    archive.error2_velocity_archive[i] = controller.error_velocity[1];

    controller.r[0] = controller.error_velocity[0] + controller.Lambda[0][0] * controller.error[0];  // filtered tracking error of link 1, defined in Eq. 34
    controller.dqr[0] = dq1_desired.y[i] + controller.Lambda[0][0] * controller.error[0];            // derivative of filtered tracking error of link 1
    controller.ddqr[0] = ddq1_desired.y[i] + controller.Lambda[0][0] * controller.error_velocity[0]; // second order derivative of filtered tracking error of link 1
    controller.r[1] = controller.error_velocity[1] + controller.Lambda[1][1] * controller.error[1];  // filtered tracking error of link 2, defined in Eq. 34
    controller.dqr[1] = dq2_desired.y[i] + controller.Lambda[1][1] * controller.error[1];            // derivative of filtered tracking error of link 2
    controller.ddqr[1] = ddq2_desired.y[i] + controller.Lambda[1][1] * controller.error_velocity[1]; // second order derivative of filtered tracking error of link 2
    // printf("controller.ddqr[1] = %f\n", controller.ddqr[1]);

    dynamics.M[0][0] = p[0] + p[1] + 2 * p[2] * cos(system_state.q2);
    dynamics.M[0][1] = p[1] + p[2] * cos(system_state.q2);
    dynamics.M[1][0] = p[1] + p[2] * cos(system_state.q2);
    dynamics.M[1][1] = p[1];

    dynamics.Vm[0][0] = -p[2] * system_state.dq2 * sin(system_state.q2);
    dynamics.Vm[0][1] = -p[2] * (system_state.dq1 + system_state.dq2) * system_state.q2;
    dynamics.Vm[1][0] = p[2] * system_state.dq1 * sin(system_state.q2);
    dynamics.Vm[1][1] = 0;

    dynamics.G[0] = p[3] * g * cos(system_state.q1) + p[4] * g * cos(system_state.q1 + system_state.q2);
    dynamics.G[1] = p[4] * g * cos(system_state.q1 + system_state.q2);
    // printf("dynamics.G[1] = %f\n", dynamics.G[1]);

    controller.controller_output[0] = dynamics.M[0][0] * controller.ddqr[0] + dynamics.M[0][1] * controller.ddqr[1] + dynamics.Vm[0][0] * controller.dqr[0] + dynamics.Vm[0][1] * controller.dqr[1] + dynamics.G[0] + controller.Kv[0] * controller.r[0]; // controller neural network's input
    controller.controller_output[1] = dynamics.M[1][0] * controller.ddqr[0] + dynamics.M[1][1] * controller.ddqr[1] + dynamics.Vm[1][0] * controller.dqr[0] + dynamics.Vm[1][1] * controller.dqr[1] + dynamics.G[1] + controller.Kv[1] * controller.r[1];

    controller.controller_out[0] = controller.controller_output[0];
    controller.controller_out[1] = controller.controller_output[1];
    controller.controller_out[2] = controller.r[0];
    controller.controller_out[3] = controller.r[1];
    archive.controller_output1_archive[i] = controller.controller_out[0];
    archive.controller_output2_archive[i] = controller.controller_out[1];
}

struct _plant{
    double plant_u[2];
    double plant_out[4];
} plant;

void PLANT_init(){
    plant.plant_u[0] = 0.0;
    plant.plant_u[1] = 0.0;
    plant.plant_out[0] = system_state.q1;
    plant.plant_out[1] = system_state.dq1;
    plant.plant_out[2] = system_state.q2;
    plant.plant_out[3] = system_state.dq2;
}

double COMPENSATOR_realize(int i){
    compensator.compensator_u[0] = controller.controller_out[0]; // controller neural network's input
    compensator.compensator_u[1] = controller.controller_out[1];
    compensator.compensator_u[2] = controller.controller_out[2]; // filtered tracking error of link 1, defined in Eq. 34
    compensator.compensator_u[3] = controller.controller_out[3]; // filtered tracking error of link 2, defined in Eq. 34

    // neural network compensator output equals neural network controller output plus neural network output, Eq. 13
    compensator.compensator_output[0] = compensator.compensator_u[0] + controller_NN_structure.neural_output[0];
    compensator.compensator_output[1] = compensator.compensator_u[1] + controller_NN_structure.neural_output[1];
    archive.compensator_output1_archive[i] = compensator.compensator_output[0];
    archive.compensator_output2_archive[i] = compensator.compensator_output[1];
    // printf("compensator.compensator_output[1] = %f\n", compensator.compensator_output[1]);

    // compensator neural network
    compensator_NN_structure.input[0] = compensator.compensator_output[0]; // compensator neural network's input
    compensator_NN_structure.input[1] = compensator.compensator_output[1];

    // compensator neural network hidden layer's output
    for (int j = 0; j < H; j++){
        compensator_NN_structure.hidden_output[0][j] = exp((-pow(compensator_NN_structure.input[0] - center_compensator[0][j], 2)) / (width_compensator * width_compensator));
        compensator_NN_structure.hidden_output[1][j] = exp((-pow(compensator_NN_structure.input[1] - center_compensator[1][j], 2)) / (width_compensator * width_compensator));
    }

    // compensator neural network's hidden layer output's derivative
    for (int j = 0; j < H; j++){
        compensator_NN_structure.hidden_output_derivative[0][j] = compensator_NN_structure.hidden_output[0][j] * (2 * (compensator_NN_structure.input[0] - center_compensator[0][j]) / (width_compensator * width_compensator));
        compensator_NN_structure.hidden_output_derivative[1][j] = compensator_NN_structure.hidden_output[1][j] * (2 * (compensator_NN_structure.input[1] - center_compensator[1][j]) / (width_compensator * width_compensator));
    }

    // for (int j = 0; j < H; j++) {
    //     printf("compensator_NN_structure.hidden_output[0][%d] = %f\n", j, compensator_NN_structure.hidden_output[0][j]);
    // }

    // controller neural network
    controller_NN_structure.input[0] = compensator.compensator_u[0]; // controller neural network's input
    controller_NN_structure.input[1] = compensator.compensator_u[1];

    // compensator neural network hidden layer's output
    for (int j = 0; j < H; j++){
        controller_NN_structure.hidden_output[0][j] = exp((-pow(controller_NN_structure.input[0] - center_controller[0][j], 2)) / (width_controller * width_controller));
        controller_NN_structure.hidden_output[1][j] = exp((-pow(controller_NN_structure.input[1] - center_controller[1][j], 2)) / (width_controller * width_controller));
    }

    for (int j = 0; j < H; j++){
        compensator_NN_structure.weight_derivative[0][j] = 0.0;
        compensator_NN_structure.weight_derivative[1][j] = 0.0;
    }

    for (int j = 0; j < H; j++){
        controller_NN_structure.weight_derivative[0][j] = 0.0;
        controller_NN_structure.weight_derivative[1][j] = 0.0;
    }

    // network tuning algorithm, Eq. 32
    // compensator neural network's weight's derivative
    for (int j = 0; j < H; j++){
        for (int k = 0; k < H; k++){
            compensator_NN_structure.weight_derivative[0][j] += -compensator.S * compensator_NN_structure.hidden_output_derivative[0][j] * controller_NN_structure.weight[0][k] * controller_NN_structure.hidden_output[0][k] * compensator.compensator_u[2];
        }
    }

    for (int j = 0; j < H; j++){
        compensator_NN_structure.weight_derivative[0][j] -= compensator.k1 * compensator.S * sqrt(pow(compensator.compensator_u[2], 2) + pow(compensator.compensator_u[3], 2)) * compensator_NN_structure.weight[0][j];
    }

    for (int j = 0; j < H; j++){
        for (int k = 0; k < H; k++){
            compensator_NN_structure.weight_derivative[1][j] += -compensator.S * compensator_NN_structure.hidden_output_derivative[1][j] * controller_NN_structure.weight[1][k] * controller_NN_structure.hidden_output[1][k] * compensator.compensator_u[3];
        }
    }

    for (int j = 0; j < H; j++){
        compensator_NN_structure.weight_derivative[1][j] -= compensator.k1 * compensator.S * sqrt(pow(compensator.compensator_u[2], 2) + pow(compensator.compensator_u[3], 2)) * compensator_NN_structure.weight[1][j];
    }

    // for (int j = 0; j < H; j++){
    //     printf("compensator_NN_structure.weight_derivative[1][%d] = %f\n", j, compensator_NN_structure.weight_derivative[1][j]);
    // }

    // controller neural network's weight's derivative
    for (int j = 0; j < H; j++){
        for (int k = 0; k < H; k++){
            controller_NN_structure.weight_derivative[0][j] += compensator.T * controller_NN_structure.hidden_output[0][j] * compensator_NN_structure.weight[0][k] * compensator_NN_structure.hidden_output_derivative[0][k] * compensator.compensator_u[2];
        }
    }

    for (int j = 0; j < H; j++){
        controller_NN_structure.weight_derivative[0][j] -= compensator.k1 * compensator.T * compensator.compensator_u[2] * controller_NN_structure.weight[0][j];
    }

    for (int j = 0; j < H; j++){
        controller_NN_structure.weight_derivative[0][j] -= compensator.k2 * compensator.T * compensator.compensator_u[2] * frobenius_norm(H, controller_NN_structure.weight[0]) * controller_NN_structure.weight[0][j];
    }

    for (int j = 0; j < H; j++){
        for (int k = 0; k < H; k++){
            controller_NN_structure.weight_derivative[1][j] += compensator.T * controller_NN_structure.hidden_output[1][j] * compensator_NN_structure.weight[1][k] * compensator_NN_structure.hidden_output_derivative[1][k] * compensator.compensator_u[3];
        }
    }

    for (int j = 0; j < H; j++){
        controller_NN_structure.weight_derivative[1][j] -= compensator.k1 * compensator.T * compensator.compensator_u[3] * controller_NN_structure.weight[1][j];
    }

    for (int j = 0; j < H; j++){
        controller_NN_structure.weight_derivative[1][j] -= compensator.k2 * compensator.T * compensator.compensator_u[3] * frobenius_norm(H, controller_NN_structure.weight[1]) * controller_NN_structure.weight[1][j];
    }

    for (int j = 0; j < H; j++){
        compensator_NN_structure.weight[0][j] += compensator_NN_structure.weight_derivative[0][j] * Ts;
        compensator_NN_structure.weight[1][j] += compensator_NN_structure.weight_derivative[1][j] * Ts;
    }

    for (int j = 0; j < H; j++){
        controller_NN_structure.weight[0][j] += controller_NN_structure.weight_derivative[0][j] * Ts;
        controller_NN_structure.weight[1][j] += controller_NN_structure.weight_derivative[1][j] * Ts;
    }

    // controller neural network's output
    controller_NN_structure.neural_output[0] = 0.0;
    controller_NN_structure.neural_output[1] = 0.0;

    for (int j = 0; j < H; j++){
        controller_NN_structure.neural_output[0] += controller_NN_structure.weight[0][j] * controller_NN_structure.hidden_output[0][j];
        controller_NN_structure.neural_output[1] += controller_NN_structure.weight[1][j] * controller_NN_structure.hidden_output[1][j];
    }
    // printf("controller_NN_structure.neural_output[1] = %f\n", controller_NN_structure.neural_output[1]);

    // neural network compensator output equals neural network controller output plus neural network output, Eq. 13
    compensator.compensator_output[0] = compensator.compensator_u[0] + controller_NN_structure.neural_output[0];
    compensator.compensator_output[1] = compensator.compensator_u[1] + controller_NN_structure.neural_output[1];

    archive.compensator_output1_archive[i] = compensator.compensator_output[0];
    archive.compensator_output2_archive[i] = compensator.compensator_output[1];

    // deadzone, assume that functions h(u) and g(u) are linearly monotonically increasing
    for (int j = 0; j < n_joint; j++){
        if (compensator.compensator_output[j] <= deadzone.start_deadzone){
            torque[j] = compensator.compensator_output[j] + deadzone.start_deadzone;
        }
        else if (compensator.compensator_output[j] >= deadzone.end_deadzone){
            torque[j] = compensator.compensator_output[j] - deadzone.end_deadzone;
        }
        else{
            torque[j] = 0.0;
        }
    }

    // for (int j = 0; j < n_joint; j++) {
    //     printf("torque[%d] = %f\n", j, torque[j]);
    // }

    archive.torque1_archive[i] = torque[0];
    archive.torque2_archive[i] = torque[1];

    // control torque equal to compensator output plus saturation function
    compensator.sat_function[0] = compensator.compensator_output[0] - torque[0];
    compensator.sat_function[1] = compensator.compensator_output[1] - torque[1];
    archive.sat_function1_archive[i] = compensator.sat_function[0];
    archive.sat_function2_archive[i] = compensator.sat_function[1];

    compensator.compensator_out[0] = torque[0];
    compensator.compensator_out[1] = torque[1];
}

double PLANT_realize(int i){
    plant.plant_u[0] = compensator.compensator_out[0];
    plant.plant_u[0] = compensator.compensator_out[1];

    double inv_M[n_joint][n_joint]; // inertia matrix's transpose
    inv_matrix(inv_M, dynamics.M, n_joint);

    double to1_Vmdq_G1, to1_Vmdq_G2;
    to1_Vmdq_G1 = torque[0] - (dynamics.Vm[0][0] * system_state.dq1 + dynamics.Vm[0][1] * system_state.dq2) - dynamics.G[0];
    to1_Vmdq_G2 = torque[1] - (dynamics.Vm[1][0] * system_state.dq1 + dynamics.Vm[1][1] * system_state.dq2) - dynamics.G[1];

    system_state.ddq1 = inv_M[0][0] * to1_Vmdq_G1 + inv_M[0][1] * to1_Vmdq_G2; // manipulator dynamics equation, Eq. 1
    system_state.ddq2 = inv_M[1][0] * to1_Vmdq_G1 + inv_M[1][1] * to1_Vmdq_G2;
    system_state.dq1 = system_state.dq1 + system_state.ddq1 * Ts;
    system_state.dq2 = system_state.dq2 + system_state.ddq2 * Ts;
    system_state.q1 = system_state.q1 + system_state.dq1 * Ts;
    system_state.q2 = system_state.q2 + system_state.dq2 * Ts;

    archive.q1_archive[i] = system_state.q1;
    archive.dq1_archive[i] = system_state.dq1;
    archive.q2_archive[i] = system_state.q2;
    archive.dq2_archive[i] = system_state.dq2;
}

void saveArchiveToTxt(double *archive, int size, const char *filename){

    FILE *file = fopen(filename, "w+");

    if (file == NULL){
        perror("Failed to open file");
        exit(1);
    }
    else{
        for (int i = 0; i < size; i++){
            fprintf(file, "%lf\n", archive[i]);
        }
        fclose(file);
        printf("Saved to file %s\n", filename);
    }
}

void saveArchive(){

    saveArchiveToTxt(q1_desired.y, ARRAY_SIZE, "../report/qd1.txt");
    saveArchiveToTxt(archive.q1_archive, ARRAY_SIZE, "../report/q1.txt");
    saveArchiveToTxt(archive.dq1_archive, ARRAY_SIZE, "../report/dq1.txt");
    saveArchiveToTxt(q2_desired.y, ARRAY_SIZE, "../report/qd2.txt");
    saveArchiveToTxt(archive.q2_archive, ARRAY_SIZE, "../report/q2.txt");
    saveArchiveToTxt(archive.dq2_archive, ARRAY_SIZE, "../report/dq2.txt");
    saveArchiveToTxt(archive.error1_archive, ARRAY_SIZE, "../report/error1.txt");
    saveArchiveToTxt(archive.error1_velocity_archive, ARRAY_SIZE, "../report/error1_velocity.txt");
    saveArchiveToTxt(archive.error2_archive, ARRAY_SIZE, "../report/error2.txt");
    saveArchiveToTxt(archive.error2_velocity_archive, ARRAY_SIZE, "../report/error2_velocity.txt");
    saveArchiveToTxt(archive.controller_output1_archive, ARRAY_SIZE, "../report/controller_output1.txt");
    saveArchiveToTxt(archive.controller_output2_archive, ARRAY_SIZE, "../report/controller_output2.txt");
    saveArchiveToTxt(archive.compensator_output1_archive, ARRAY_SIZE, "../report/compensator_output1.txt");
    saveArchiveToTxt(archive.compensator_output2_archive, ARRAY_SIZE, "../report/compensator_output2.txt");
    saveArchiveToTxt(archive.torque1_archive, ARRAY_SIZE, "../report/torque1.txt");
    saveArchiveToTxt(archive.torque2_archive, ARRAY_SIZE, "../report/torque2.txt");
    saveArchiveToTxt(archive.sat_function1_archive, ARRAY_SIZE, "../report/sat_function1.txt");
    saveArchiveToTxt(archive.sat_function2_archive, ARRAY_SIZE, "../report/sat_function2.txt");
}

int main(){

    SystemInput(&q1_desired, &dq1_desired, &ddq1_desired, &q2_desired, &dq2_desired, &ddq2_desired, Ts, t0, t1);
    CONTROLLER_init();  // initialize controller parameter
    COMPENSATOR_init(); // initialize neural network compensator
    PLANT_init();       // initialize plant parameter

    for (int i = 0; i < ARRAY_SIZE; i++){
    // for (int i = 0; i < 200; i++){
        double time = i * Ts + t0;
        printf("time at step %d: %f\n", i, time);
        CONTROLLER_realize(i);
        COMPENSATOR_realize(i);
        PLANT_realize(i);
    }

    saveArchive();

    return 0;
}
