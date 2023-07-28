from prediction.pred_util import *
import torch.multiprocessing
from recording.util import *
from recording.util import set_subframe
from pose.mediapipe_minimal import MediaPipeWrapper


torch.multiprocessing.set_sharing_strategy('file_system')

disp_x = int(1920 * 1)
disp_y = int(1080 * 1)
aspect_ratio = 1920 / 1080


def get_full_frame_1d(cropped_pressure_img, pose_bbox, destination_size=None):
    if destination_size is None:
        full_frame_pressure = np.zeros((1080, 1920), dtype=np.uint8)
    else:
        full_frame_pressure = np.zeros((destination_size[0], destination_size[1]), dtype=np.uint8)
    max_x = int(pose_bbox['max_x'])
    min_x = int(pose_bbox['min_x'])
    max_y = int(pose_bbox['max_y'])
    min_y = int(pose_bbox['min_y'])

    resize_dims = (max_x - min_x, max_y-min_y)
    resized_pressure = cv2.resize(cropped_pressure_img, resize_dims)
    full_frame_pressure[min_y:max_y, min_x:max_x] = resized_pressure
    return full_frame_pressure


def run_model(img, model, config, zero_downweight=1):
    with torch.no_grad():
        if config.FORCE_CLASSIFICATION:
            force_pred_class = model(img.cuda())
            if isinstance(force_pred_class, tuple):
                force_pred_class = force_pred_class[0]

            force_pred_class = torch.argmax(force_pred_class, dim=1)
            force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
        else:
            force_pred_scalar = model(img.cuda()).squeeze(1) * config.NORM_FORCE_REGRESS

    return force_pred_scalar.detach()


def process_and_run_model(img, best_model, config):
    # Takes in a cropped image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255
    # img = seq_reader.crop_img(img, 0, config)
    img = resnet_preprocessor(img)
    img = img.transpose(2, 0, 1).astype('float32')
    img = torch.tensor(img).unsqueeze(0)

    force_pred_scalar = run_model(img, best_model, config)

    force_pred_scalar = force_pred_scalar.detach().cpu().squeeze().numpy()

    # force_color_pred = pressure_to_colormap(force_pred_scalar)
    return force_pred_scalar


def get_hand_bbox(img, mp_wrapper):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_wrapper.run_img(img, vis=False)
    scale = 1.5

    hand_joints = result['right_points']
    if result['left_points'] is not None:
        hand_joints = result['left_points']

    if hand_joints is not None:
        hand_joints = np.array(hand_joints)
        hand_joints[:, 0] *= img.shape[1]
        hand_joints[:, 1] *= img.shape[0]

        center_x = (hand_joints[:, 0].min() + hand_joints[:, 0].max()) / 2
        center_y = (hand_joints[:, 1].min() + hand_joints[:, 1].max()) / 2

        radius = max(hand_joints[:, 0].max() - center_x, hand_joints[:, 1].max() - center_y)
        radius = radius * scale
    else:
        center_x = 960
        center_y = 540
        radius = 10000

    out_dict = dict()
    out_dict['min_x'] = max(0, center_x - radius)
    out_dict['max_x'] = min(img.shape[1], center_x + radius)
    out_dict['min_y'] = max(0, center_y - radius)
    out_dict['max_y'] = min(img.shape[0], center_y + radius)

    for key, value in out_dict.items():
        out_dict[key] = int(round(value))

    return out_dict


def draw_bbox_full_frame(img, pose_bbox):
    max_x = int(pose_bbox['max_x'])  # draw crop bbox
    min_x = int(pose_bbox['min_x'])
    max_y = int(pose_bbox['max_y'])
    min_y = int(pose_bbox['min_y'])

    cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 0, 255), 4)
    cv2.line(img, (min_x, max_y), (max_x, max_y), (0, 0, 255), 4)
    cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 0, 255), 4)
    cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 0, 255), 4)


def test():
    window_name = 'Weak Label Demo'
    disp_frame = np.zeros((disp_y, disp_x, 3), dtype=np.uint8)
    mp_wrapper = MediaPipeWrapper()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Turn the autofocus off
    cap.set(cv2.CAP_PROP_FOCUS, 15)  # Set to predetermined focus value for BRIO

    best_model = torch.load(find_latest_checkpoint(config.CONFIG_NAME))
    best_model.eval()

    while True:
        ret, camera_frame = cap.read()
        if camera_frame is None:
            continue

        base_img = camera_frame

        bbox = get_hand_bbox(base_img, mp_wrapper)
        crop_frame = base_img[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x'], ...]
        crop_frame = cv2.resize(crop_frame, (config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y))    # image is YX

        force_pred = process_and_run_model(crop_frame, best_model, config) * 2
        force_pred_full_frame = get_full_frame_1d(force_pred, bbox)
        force_pred_color_full_frame = pressure_to_colormap(force_pred_full_frame)

        overlay_frame = cv2.addWeighted(base_img, 0.6, force_pred_color_full_frame, 1.0, 0.0)
        draw_bbox_full_frame(overlay_frame, bbox)

        set_subframe(0, overlay_frame, disp_frame, steps_x=1, steps_y=1, title='Network Output with Overlay')

        cv2.imshow(window_name, disp_frame)
        keycode = cv2.waitKey(1) & 0xFF

        if keycode == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test()
