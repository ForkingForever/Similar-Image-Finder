

# loop over the image hashes
for (h, hashedPaths) in hashes.items():
    # check to see if there is more than one image with the same hash
    if len(hashedPaths) > 1:
        # check to see if this is a dry run
        if args["remove"] <= 0:
            # initialize a montage to store all images with the same
            # hash
            montage = None

            # loop over all image paths with the same hash
            for p in hashedPaths:
                # load the input image and resize it to a fixed width
                # and height
                image = cv2.imread(p)
                image = cv2.resize(image, (150, 150))

                # if our montage is None, initialize it
                if montage is None:
                    montage = image

                # otherwise, horizontally stack the images
                else:
                    montage = np.hstack([montage, image])

            # show the montage for the hash
            print("[INFO] hash: {}".format(h))
            cv2.imshow("Montage", montage)
            cv2.waitKey(0)

        # otherwise, we'll be removing the duplicate images
        else:
            # loop over all image paths with the same hash *except*
            # for the first image in the list (since we want to keep
            # one, and only one, of the duplicate images)
            for p in hashedPaths[1:]:
                os.remove(p)
