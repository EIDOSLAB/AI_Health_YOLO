def split_data(train_img, train_label):
    test_img = train_img[-36:]
    test_label = []
    for i in range(len(train_label)):
        test_label.append(train_label[i][-36:])
        print("shape of testing label[%s]:" % i, test_label[i].shape)
    print("shape of testing img:", test_img.shape)
    print()

    valid_img = train_img[296:332]
    valid_label = []
    for i in range(len(train_label)):
        valid_label.append(train_label[i][296:332])
        print("shape of validation label[%s]:" % i, valid_label[i].shape)
    print("shape of validation img:", valid_img.shape)
    print()

    train_img = train_img[:296]
    new_train_label = []
    for i in range(len(train_label)):
        new_train_label.append(train_label[i][:296])
        print("shape of training label[%s]:" % i, new_train_label[i].shape)
    train_label = new_train_label
    print("shape of training img:", train_img.shape)

    return (train_img, train_label), (valid_img, valid_label), (test_img, test_label)