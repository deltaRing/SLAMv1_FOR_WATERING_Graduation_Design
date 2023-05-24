#ifndef _DBOW_TRAIN_H_
#define _DBOW_TRAIN_H_

#include <DBoW3/DBoW3.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "../datapath/datapath.h"
#include "../mapping/posture.h"
#include "../define/map.h"
#include <termios.h>
#include <dirent.h>
using namespace cv;
using namespace std;

void feature_training();

void load_vocab(bool train=false); // load weight of vacab
void compute_vocab(pFrame kf1); // compute vocab
double score_vocab(pFrame kf1, pFrame kf2); // get similarity of keyframes
double score_vocab(DBoW3::BowVector bv1, DBoW3::BowVector bv2);

#endif
