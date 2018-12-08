def data_generator(images, labels, batch_size=2, dim=(768, 768), n_classes=2, shuffle=True):
    # Initialization
    data_size = len(images)
    nbatches = data_size // batch_size
    list_IDs = np.arange(data_size)
    indices = list_IDs
    
    # Data generation
    while True:
        if shuffle == True:
            np.random.shuffle(indices)
        for index in range(nbatches):
            batch_indices = indices[index*batch_size:(index+1)*batch_size]

            X = np.empty((batch_size, *dim, 3))
            y_semseg = np.empty((batch_size, *dim), dtype=int)

            for i, ID in enumerate(batch_indices):
                image = cv2.resize(np.array(imageio.imread(images[ID]), dtype=np.uint8), dim[1::-1])
                label = cv2.resize(imageio.imread(labels[ID]), dim[1::-1], interpolation=cv2.INTER_NEAREST)
                X[i,] = image
                y_semseg[i] = label
            yield (preprocess_input(X), to_categorical(y_semseg, num_classes=n_classes))