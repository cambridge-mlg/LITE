from meta_dataset_reader import MetaDatasetReader


def get_dataset_reader(args, train_set, validation_set, test_set):
    dataset = MetaDatasetReader(
        data_path=args.data_path,
        mode=args.mode,
        train_set=train_set,
        validation_set=validation_set,
        test_set=test_set,
        max_way_train=args.max_way_train,
        max_way_test=50,
        max_support_train=args.max_support_train,
        max_support_test=500,
        max_query_train=10,
        max_query_test=10,
        image_size=args.image_size)

    return dataset
