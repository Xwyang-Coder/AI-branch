
#include "simple_yolo.hpp"
#include "nvToolsExt.h"

#if defined(_WIN32)
#	include <Windows.h>
#   include <wingdi.h>
#	include <Shlwapi.h>
#	pragma comment(lib, "shlwapi.lib")
#   pragma comment(lib, "ole32.lib")
#   pragma comment(lib, "gdi32.lib")
#	undef min
#	undef max
#else
#	include <dirent.h>
#	include <sys/types.h>
#	include <sys/stat.h>
#	include <unistd.h>
#   include <stdarg.h>
#endif

using namespace std;

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p; break;
    case 1:r = q; g = v; b = p; break;
    case 2:r = p; g = v; b = t; break;
    case 3:r = p; g = q; b = v; break;
    case 4:r = t; g = p; b = v; break;
    case 5:r = v; g = p; b = q; break;
    default:r = 1; g = 1; b = 1; break;
    }
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static bool exists(const string& path) {

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

static string get_file_name(const string& path, bool include_suffix) {

    if (path.empty()) return "";

    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;

    //include suffix
    if (include_suffix)
        return path.substr(p);

    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}

static double timestamp_now_float() {
    return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

bool requires_model(const string& name) {

    auto onnx_file = cv::format("%s.onnx", name.c_str());
    //std::cout << "Checking existence of: " << onnx_file << std::endl;

    if (!exists(onnx_file)) {
        printf("Auto download %s\n", onnx_file.c_str());
        //system(cv::format("wget http://zifuture.com:1556/fs/25.shared/%s", onnx_file.c_str()).c_str());
    }

    bool isexists = exists(onnx_file);
    if (!isexists) {
        printf("Download %s failed\n", onnx_file.c_str());
    }
    return isexists;
}

static void inference_and_performance(int deviceid, const string& engine_file, SimpleYolo::Mode mode, SimpleYolo::Type type, const string& model_name, cv::Mat frame, int cur_f, cv::VideoWriter video_mask) {

    auto engine = SimpleYolo::create_infer(engine_file, type, deviceid);
    if (engine == nullptr) {
        printf("Engine is nullptr\n");
    }

    int w = 1920;
    int h = 1080;

    auto type_name = SimpleYolo::type_name(type);
    auto mode_name = SimpleYolo::mode_string(mode);
    vector<cv::Mat> images;
    cv::Size dsize = cv::Size(w, h);
    for (int i = 0; i < 1; ++i) {
        cv::resize(frame, frame, dsize);
        images.emplace_back(frame);
    }

    vector<shared_future<SimpleYolo::SkinArray>> skin_mask_array;

	// warmup
	for (int i = 0; i < 3; ++i)
        skin_mask_array = engine->commits(images);
    skin_mask_array.back().get();
    skin_mask_array.clear();
    
    /*
    while (1) {
        auto begin_timer = timestamp_now_float();
        skin_mask_array = engine->commits(images);

        // wait all result
        skin_mask_array.back().get();
        float inference_average_time = (timestamp_now_float() - begin_timer) / images.size();
        //printf("[%-4d] The times of [ SkinSegment ] for an image is: [image, FPS] => [%.2f ms, %.2f]\n", cur_f, inference_average_time, 1000 / inference_average_time);
        printf("The times of [ SkinSegment_%d ] for an image is: [image, FPS] => [%.2f ms, %.2f]\n", mode, inference_average_time, 1000 / inference_average_time);
    }
    */

    
    auto begin_timer = timestamp_now_float();
    nvtxRangePushA("engine->commits");
    skin_mask_array = engine->commits(images);
    nvtxRangePop();
    // wait all result
    skin_mask_array.back().get();
    float inference_average_time = (timestamp_now_float() - begin_timer) / images.size();
    printf("[%-4d] The times of [ SkinSegment_%d ] for an image is: [image, FPS] => [%.2f ms, %.2f]\n", cur_f, mode, inference_average_time, 1000 / inference_average_time);

    //std::cout << "====skin_mask_array====: " << skin_mask_array.size() << std::endl;
    for (int i = 0; i < skin_mask_array.size(); ++i) {
        // 获取实际的 SkinArray 结果
        SimpleYolo::SkinArray skin_array = skin_mask_array[i].get();

        // 遍历 SkinArray 中的每个 holly_skin 对象
        for (const auto& skin : skin_array) {
            cv::Mat mask_roi(480, 480, CV_8UC1, const_cast<uint8_t*>(skin.skin_mask));
            cv::resize(mask_roi, mask_roi, cv::Size(w, h));
            //cv::imwrite("./mask.jpg", mask_roi);

            //cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            //video.write(frame);
            cv::cvtColor(mask_roi, mask_roi, cv::COLOR_GRAY2BGR);
            video_mask.write(mask_roi);
        }
    }
    
    engine.reset();
}

static void test(SimpleYolo::Type type, SimpleYolo::Mode mode, const string& model) {

    int deviceid = 0;
    auto mode_name = SimpleYolo::mode_string(mode);
    SimpleYolo::set_device(deviceid);
    const char* name = model.c_str();

    int cur_f = 0;
    int w = 1920;
    int h = 1080;
    cv::VideoWriter video;
    cv::VideoWriter video_mask;
    cv::VideoCapture capture("E:\\vs_project\\My_skin_segment\\workspace\\images\\test3_1.mp4");
    std::string result_video = "E:\\vs_project\\My_skin_segment\\workspace\\images\\out_skinout.avi";
    std::string result_video_mask = "E:\\vs_project\\My_skin_segment\\workspace\\images\\out_skinout_mask.avi";

    printf("===================== test %s %s %s =====================\n", SimpleYolo::type_name(type), mode_name, name);

    //if (!requires_model(name))
        //return;

    string onnx_file = cv::format("%s.onnx", name);
    string model_file = cv::format("%s_%s.engine", name, mode_name);  // skin_segment_FP32_FP32
    //string model_file = cv::format("%s_%s.bin", name, mode_name);  // yolov5s_FP32

    int test_batch_size = 1;

    if (!exists(model_file)) {
        SimpleYolo::compile(
            mode,                 // FP32、FP16、INT8
            type,                 // Skin ...
            test_batch_size,      // max batch size
            onnx_file,            // source
            model_file,           // save to
            1 << 30,
            "E:\\vs_project\\My_skin_segment\\workspace\\int8_images",
            "E:\\vs_project\\My_skin_segment\\workspace\\int8_images_entropy\\entropy.cache"
        );
    }

    if (!capture.isOpened()) {
        printf("could not read this video file...\n");
        return;
    }

    while (true) {
        cv::Mat frame;
        if (!capture.read(frame)) {
            std::cout << "\n1.Cannot read the video file, please check your video. (or 2.the video file is complete!).\n";
            break;
        }

        if (cur_f == 0) {
            //video.open(result_video, cv::VideoWriter::fourcc('M','J','P','G'), 30, frame.size());		// bool:true = iscolor
            video_mask.open(result_video_mask, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(w, h)); // Changed encoding format, bool:false = nocolor
        }

        cur_f++;
        inference_and_performance(deviceid, model_file, mode, type, name, frame, cur_f, video_mask);

        //cv::imwrite("E:\\vs_project\\My_skin_segment\\workspace\\results\\video_images\\Src_1920_1080.jpg", frame);
        //break;
    }

    std::cout << "Total_frames:" << cur_f << std::endl;

    capture.release();
    video_mask.release();
    
}

static void test_dptv2(SimpleYolo::Type type, SimpleYolo::Mode mode, const string& model) {

	int deviceid = 0;
	auto mode_name = SimpleYolo::mode_string(mode);
	SimpleYolo::set_device(deviceid);
	const char* name = model.c_str();

	int cur_f = 0;
	int w = 1920;
	int h = 1080;
	cv::VideoWriter video;
	cv::VideoWriter video_mask;
	cv::VideoCapture capture("E:\\vs_project\\My_skin_segment\\workspace\\images\\test3_1.mp4");
	std::string result_video = "E:\\vs_project\\My_skin_segment\\workspace\\images\\out_skinout.avi";
	std::string result_video_mask = "E:\\vs_project\\My_skin_segment\\workspace\\images\\out_skinout_mask.avi";

	printf("===================== test %s %s %s =====================\n", SimpleYolo::type_name(type), mode_name, name);

	//if (!requires_model(name))
		//return;

	string onnx_file = cv::format("%s.onnx", name);
	string model_file = cv::format("%s_%s.trtmodel", name, mode_name);  // skin_segment_FP32_FP32
	//string model_file = cv::format("%s_%s.bin", name, mode_name);  // yolov5s_FP32

	int test_batch_size = 1;

	if (!exists(model_file)) {
		SimpleYolo::compile(
			mode,                 // FP32、FP16、INT8
			type,                 // Skin ...
			test_batch_size,      // max batch size
			onnx_file,            // source
			model_file,           // save to
			1 << 30,
			"",
			""
		);
	}

	if (!capture.isOpened()) {
		printf("could not read this video file...\n");
		return;
	}

	while (true) {
		cv::Mat frame;
		if (!capture.read(frame)) {
			std::cout << "\n1.Cannot read the video file, please check your video. (or 2.the video file is complete!).\n";
			break;
		}

		if (cur_f == 0) {
			//video.open(result_video, cv::VideoWriter::fourcc('M','J','P','G'), 30, frame.size());		// bool:true = iscolor
			video_mask.open(result_video_mask, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(w, h)); // Changed encoding format, bool:false = nocolor
		}

		cur_f++;
		inference_and_performance(deviceid, model_file, mode, type, name, frame, cur_f, video_mask);

		//cv::imwrite("E:\\vs_project\\My_skin_segment\\workspace\\results\\video_images\\Src_1920_1080.jpg", frame);
		//break;
	}

	std::cout << "Total_frames:" << cur_f << std::endl;

	capture.release();
	video_mask.release();
}


int main() {

    test(SimpleYolo::Type::Skin, SimpleYolo::Mode::FP16, "E:\\vs_project\\My_skin_segment\\workspace\\models\\skin_segment_0820");
    //test(SimpleYolo::Type::Skin, SimpleYolo::Mode::FP16, "E:\\vs_project\\My_skin_segment\\workspace\\models\\skin_segment");
    //test(SimpleYolo::Type::Skin, SimpleYolo::Mode::INT8, "E:\\vs_project\\My_skin_segment\\workspace\\models\\skin_segment");
    //test_dptv2(SimpleYolo::Type::Dpt, SimpleYolo::Mode::FP32, "E:\\vs_project\\My_skin_segment\\workspace\\models\\depthv2");

    return 0;
}