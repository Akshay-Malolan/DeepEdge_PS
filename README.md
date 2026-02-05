# DeepEdge_PS
Solution for problem statement provided by DeepEdge

I opted for a synthetic coordinate-based generation strategy rather than using pre-saved image files to overcome the memory and storage limitations of Kaggle notebooks. Loading 2,500 images simultaneously would have caused system crashes; instead, I planned to generate the images on-the-fly from the coordinates using tf.data.Dataset with lazy evaluation and .map(). This helped me create the images dynamically, generating only 32 images in memory at a given point in time (in one batch). This generates directly from the coordinate data, significantly reducing the memory usage and as well reducing the disk I/O bottlenecks, thus making the training faster. I chose a 80/20 train-test split after shuffling the entire dataset with a fixed seed for reproducibility, also ensuring random spatial distribution across both splits.


For the architecture, I chose CoordConv over a traditional CNN because standard convolutions struggle to determine absolute spatial positions. Even though there exist more complex models like CoordGate, I chose CoordConv considering its lightweight and small size of dataset, while CoordGate may cause heavy overfitting. 


I explicitly injected spatial awareness by concatenating two coordinate channels (x_map and y_map) to the input image using create_coord_maps(), which generates 50×50 grids containing normalized x and y positions (0.0 to 1.0). This transforms the input from (50,50,1) to (50,50,3), where the first channel shows "where is the white pixel" and the additional channels tell the network "where am I in absolute space." then use two lightweight Conv2D layers (16 and 32 filters) to process these position-aware features, followed by GlobalMaxPooling2D which collapses spatial dimensions by taking maximum values, critically, the single white pixel (value 1.0) dominates this pooling, and when combined with the coordinate channels, the network learns to associate the max activation location with its corresponding coordinate values. Finally, a Dense layer with 2 linear outputs directly predicts the (x,y) coordinates as continuous regression values.

Reference for CoordConv – Uber Research - https://arxiv.org/pdf/1807.03247



