class Config:
    class Annotator:
        # Dataset
        img_path = "AidData/images/train"
        labels_path = "AidData/labels/train.json"
        session = 9
        # interesting_session_indices = [9, 6, 11, 7, 10, 12, 15, 13, 14]  # Train

        # Input
        load_lobsters = True
        input_base_path = "annotations/new"
        input_file_name = "a_" + str(session)

        # Output
        output_base_path = "annotations/new"
        output_file_name = "a_" + str(session)

    class TrainingDataGenerator:
        # Input
        lobster_base_path = "annotations"
        # lobs_file_names_train = [
        #     "a_0",
        #     "a_2",
        #     "a_4",
        #     "a_7",
        #     "a_9",
        #     "a_20",
        # ]
        # lobs_file_names_val = [
        #     "a_1",
        #     "a_3",
        #     "a_5",
        #     "a_6",
        #     "a_8",
        # ]
        lobs_file_names_train = [
            "a_1",
            "a_4",
            "a_5",
            "a_6",
            "a_7",
            "a_8",
            "a_9",
            "a_10",
            "a_12",
            "a_14",
            "a_15",
            "a_16",
            "a_17",
            "a_18",
            "a_19",
            "a_20",
            "a_21",
            "a_22",
            "a_24",
            "a_25",
            "a_26",
            "a_27",
            "a_28",
            "a_29",
            "a_30",
            "a_32",
            "a_33",
            "a_34",
            "a_36",
            "a_38",
            "a_39",
            "a_40",
            "a_41",
            "a_42",
        ]
        lobs_file_names_val = [
            "a_0",
            "a_2",
            "a_3",
            "a_11",
            "a_13",
            "a_23",
            "a_31",
            "a_35",
            "a_37",
        ]

        # Misc
        window_size = 5

        # Output
        output_base_path = "training_data/mars_28_kp2wireframe"
        output_file_name = "td_kps"

    class Model:
        # Input
        data_base_path = "training_data/mars_28_kp2wireframe"
        training_data_name = "td_kps"
        val_data_name = training_data_name

        # Output
        model_base_path = "models"
        model_name = "m_kps_all_data_kp2wireframe"

    class Main:
        # Input
        model_base_path = "models"
        # model_name = "m_ws10_bbs"
        model_name = "m_ws10_bbs"

        # Preditction
        window_size = 10
        minimum_distance = 200

        # Output
        video_name = "test"

        # Detection
        # model_type = "detection"
        # img_path = "data/images/test"
        # labels_path = "data/labels/test.json"
        # interesting_session_indices = [
        #     0,
        #     1,
        #     4,
        #     5,
        #     6,
        #     7,
        #     3,
        # ]

        # Fasit
        model_type = "fasit"
        img_path = "AidData/images/train"
        labels_path = "AidData/labels/train.json"
        interesting_session_indices = [12]
        # interesting_session_indices = [9, 6, 11, 7, 10, 12, 15, 13, 14]  # Train
        # interesting_session_indices = [
        #     0,
        #     1,
        #     2,
        #     3,
        #     4,
        #     5,
        #     6,
        #     7,
        #     8,
        #     9,
        #     10,
        #     11,
        #     12,
        #     13,
        #     14,
        #     15,
        #     16,
        #     17,
        #     18,
        #     19,
        #     20,
        #     21,
        #     22,
        #     23,
        #     24,
        #     25,
        #     26,
        #     27,
        #     28,
        #     29,
        #     30,
        #     31,
        #     32,
        #     33,
        #     34,
        #     35,
        #     36,
        #     37,
        #     38,
        #     39,
        #     40,
        #     41,
        #     42,
        # ]
