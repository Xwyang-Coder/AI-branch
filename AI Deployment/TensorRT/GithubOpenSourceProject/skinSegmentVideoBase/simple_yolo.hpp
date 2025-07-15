#ifndef SIMPLE_YOLO_HPP
#define SIMPLE_YOLO_HPP

/*
  �򵥵�yolo�ӿڣ����׼��ɵ��Ǹ�����
*/

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

namespace SimpleYolo {

    using namespace std;

    enum class Type : int {
        V5,
        Skin,
        Dpt
    };

    enum class Mode : int {
        FP32,
        FP16,
        INT8
    };

    typedef struct holly_skin {
        uint8_t skin_mask[480 * 480];
        holly_skin() {
            memset(&skin_mask[0], 255, 480 * 480);
        }
    }holly_skin_t;

    typedef std::vector<holly_skin_t> SkinArray;

    class Infer {
    public:
        virtual shared_future<SkinArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<SkinArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    const char* trt_version();
    const char* type_name(Type type);
    const char* mode_string(Mode type);
    void set_device(int device_id);

    /*
        ģ�ͱ���
        max batch size��Ϊ�����������batch����
        source_onnx������ֻ֧��onnx��ʽ����
        saveto�������tensorRTģ�ͣ����ں����ļ���
        max workspace size��������ռ��С��һ���1GB����Ƕ��ʽ���Ը�Ϊ256MB����λ��byte
        int8 images folder������ModeΪINT8ʱ����Ҫ�ṩͼ�����ݽ��б궨�����ṩ�ļ��У����Զ����������jpg/jpeg/tiff/png/bmp
        int8_entropy_calibrator_cache_file������int8ģʽ�£����ļ����Ի��棬������μ������ݣ����Կ�ƽ̨ʹ�ã���һ��txt�ļ�
     */
     // 1GB = 1<<30
    bool compile(
        Mode mode, Type type,
        unsigned int max_batch_size,
        const string& source_onnx,
        const string& saveto,
        size_t max_workspace_size = 1 << 30,
        const std::string& int8_images_folder = "",
        const std::string& int8_entropy_calibrator_cache_file = ""
    );

    shared_ptr<Infer> create_infer(
        const string& engine_file, 
        Type type, 
        int gpuid);

}; // namespace SimpleYolo

#endif // SIMPLE_YOLO_HPP
