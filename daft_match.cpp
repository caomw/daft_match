#include <daft/daft.h>
#include <string>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

static Mat defaultCameraMatrix()
{
    float vals[] = {525., 0., 3.1950000000000000e+02,
                    0., 525., 2.3950000000000000e+02,
                    0., 0., 1.};
    return Mat(3,3,CV_32FC1,vals).clone();
}

int main(int argc, char** argv)
{
    CV_Assert(argc == 8);

    string srcImageFilename = argv[1];
    string srcMaskFilename  = argv[2];
    string srcDepthFilename = argv[3];
    string dstImageFilename = argv[4];
    string dstMaskFilename  = argv[5];
    string dstDepthFilename = argv[6];
    bool doubleImageSize = atoi(argv[7]);

    Mat srcImage = imread(srcImageFilename, 0);
    Mat dstImage = imread(dstImageFilename, 0);
    Mat srcMask = imread(srcMaskFilename, 0);
    Mat dstMask = imread(dstMaskFilename, 0);
    Mat srcDepth = imread(srcDepthFilename, -1);
    {
        Mat depth_flt;
        srcDepth.convertTo(depth_flt, CV_32FC1, 0.001);
        depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), srcDepth == 0);
        srcDepth = depth_flt;
    }
    Mat dstDepth = imread(dstDepthFilename, -1);
    {
        Mat depth_flt;
        dstDepth.convertTo(depth_flt, CV_32FC1, 0.001);
        depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), dstDepth == 0);
        dstDepth = depth_flt;
    }

    CV_Assert(!srcImage.empty());
    CV_Assert(!dstImage.empty());
    CV_Assert(!srcMask.empty());
    CV_Assert(!dstMask.empty());
    CV_Assert(!srcDepth.empty());
    CV_Assert(!dstDepth.empty());

    // set some detector params for low-res images
    cv::daft::DAFT::DetectorParams detp;
    cv::daft::DAFT::DescriptorParams descp;

    // lower the detection threshold to get more keypoints
    detp.det_threshold_ = 0.01;

    // make the descriptor windows smaller
    descp.patch_size_ = 15;

    if ( doubleImageSize )
    {
        // double the image size
		cv::resize( srcImage, srcImage, cv::Size(), 2, 2, CV_INTER_LINEAR );
		cv::resize( dstImage, dstImage, cv::Size(), 2, 2, CV_INTER_LINEAR );
		cv::resize( srcDepth, srcDepth, cv::Size(), 2, 2, CV_INTER_NN );
		cv::resize( dstDepth, dstDepth, cv::Size(), 2, 2, CV_INTER_NN );
		cv::resize( srcMask, srcMask, cv::Size(), 2, 2, CV_INTER_NN );
		cv::resize( dstMask, dstMask, cv::Size(), 2, 2, CV_INTER_NN );
    } else {
        // allow keypoints at small scales
    	detp.min_px_scale_ = 2.0;
    }

    cv::daft::DAFT daft(detp, descp);
    vector<KeyPoint3D> srcKeypoints, dstKeypoints;
    cv::Mat1f srcDescriptors, dstDescriptors;
    Mat K = defaultCameraMatrix();

    daft(srcImage, srcMask, srcDepth, K, srcKeypoints, srcDescriptors);
    daft(dstImage, dstMask, dstDepth, K, dstKeypoints, dstDescriptors);
    std::cout << "srcKeypoints.size() == " << srcKeypoints.size() << std::endl;
    std::cout << "dstKeypoints.size() == " << dstKeypoints.size() << std::endl;

    cv::Mat srcKpImage,dstKpImage;
    cv::drawKeypoints3D( srcImage, srcKeypoints, srcKpImage, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::drawKeypoints3D( dstImage, dstKeypoints, dstKpImage, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    cv::imshow("source keypoints", srcKpImage);
    cv::imshow("destination keypoints", dstKpImage);

    vector<vector<DMatch> > allMatches;
    BFMatcher descriptorMatcher(NORM_L2SQR);
    descriptorMatcher.knnMatch(srcDescriptors, dstDescriptors, allMatches, 2);

    vector<KeyPoint> srcKeypoints2D, dstKeypoints2D;
    vector<DMatch> filteredMatches;
    for(size_t i = 0; i < allMatches.size(); i++)
    {
        const DMatch& m0 = allMatches[i][0];
        const DMatch& m1 = allMatches[i][1];

        if(m0.distance < 0.6 * m1.distance)
        {
            int ind = (int)srcKeypoints2D.size();
            srcKeypoints2D.push_back(KeyPoint(srcKeypoints[m0.queryIdx].pt, srcKeypoints[m0.queryIdx].size));
            dstKeypoints2D.push_back(KeyPoint(dstKeypoints[m0.trainIdx].pt, dstKeypoints[m0.trainIdx].size));
            filteredMatches.push_back(DMatch(ind, ind, 0));
        }
    }

    std::cout << "filteredMatches.size()== " << filteredMatches.size() << std::endl;

    Mat outImage;
    drawMatches(srcImage, srcKeypoints2D, dstImage, dstKeypoints2D, filteredMatches, outImage);
    imshow("matches", outImage);
    waitKey();
}
