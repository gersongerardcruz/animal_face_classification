# End-to-End Animal Face Classifier using Keras, Keras Tuner Mlflow, SQLite, Streamlit and FastAPI

## Project Information

This is primarily a project meant for fun and learning because who doesn't love looking at cute animals while learning?

On a more serious note, this project uses a Keras Sequential model, tuned with Keras Tuner, to classify animal faces into three categories: cat, dog, and wildlife. Images were stored locally and their file paths were stored in an SQLite database so that the training and deployment scripts can acess the database to get image file paths and feed them to the model. This is done to simulate cloud environments, where we store the locations of objects in a database like AWS RDS, while storing the actual objects in AWS S3 buckets.

To serve the model, the project uses Streamlit for the frontend connected with FastAPI in the backend. An additional feature of the project is to recommend similar images based on the prediction of the model by accessing the SQLite database and displaying images with the same label.

While this project is an educational one, I found that there is value to be derived from the use of animal classification models because are important because they have a wide range of applications in various fields such as wildlife conservation, animal behavior analysis, animal ecology, and animal identification.

## Business Value

To be more specific, here are some applications wherein businesses and organizations using animal face classification/animal classification models can derive value: 

1. Enhanced customer experience: Animal face classification can be used in industries such as pet stores, animal shelters, and zoos, to improve customer experiences. For example, a pet store can use the technology to help match customers with their perfect pet based on facial features and expressions.
2. Streamlined research and analysis: Researchers can use animal face classification models to quickly and accurately categorize large amounts of data, such as images from camera traps or satellite imagery. This can lead to more efficient and accurate research, with potential applications in fields such as ecology, biology, and conservation.
3. Improved animal welfare: Animal face classification models can help identify and monitor animal populations in the wild, aiding in conservation efforts and preventing animal poaching.
4. Enhanced wildlife conservation efforts: Animal face classification models can aid in the conservation of endangered species by allowing researchers to accurately identify and track animals in the wild. For example, the model can be used to identify individual animals and track their movements and behavior patterns, providing valuable data for conservation efforts. This can lead to improved conservation strategies and ultimately help to protect endangered species from extinction.

## Technical Value

On a more technical note, the team can generate value from utilizing Keras, Mlflow, separate locations for image paths and images, SQLite databases, and  in the development of the animal classification model because of the following:

### Keras:

- User-friendly API: Keras provides a simple and user-friendly API for building neural networks. It allows users to build a neural network model by adding layers to a Sequential model or by defining the model as a graph of layers. This makes it easier for beginners to build their own models without a deep understanding of the underlying mathematical concepts.
- Pre-trained models and transfer learning: Keras provides a variety of pre-trained models that can be fine-tuned for specific tasks using transfer learning. This allows users to take advantage of state-of-the-art models that have been trained on large datasets, without having to train them from scratch.
- Tuning: Keras models have access to KerasTuner which automates the tuning process and saves time and effort that would have been required to manually optimize the model’s performance.

### Mlflow:

- Makes it easier to reproduce experiments and results by tracking the inputs, parameters, and metrics used in the network intrusion detection model.
- Enables effective collaboration and knowledge sharing among data scientists and engineers by providing a centralized repository for model experimentation and results.
- Provides version control for models and datasets, allowing for easy comparison and selection of the best model for deployment.
- Facilitates the deployment of the network intrusion detection model by generating deployable artifacts and providing an API for serving the model.

### SQLite for storing images paths:

- Enables efficient querying and retrieval of specific images based on certain criteria (e.g. image label, date added)
- Allows for easy updating and managing of image metadata, such as image labels, descriptions, and other relevant information
- Simplifies the process of sharing and distributing image data across multiple systems and applications, as only the file paths need to be shared and not the actual images themselves

### Separate storing location for images:

- Reduces the storage requirements and costs for the database by storing the actual images in a less expensive or more scalable storage location, such as a cloud storage bucket or a distributed file system
- Improves the performance and reliability of the database by reducing the amount of data that needs to be transferred and processed by the database, allowing it to focus on managing the metadata and other aspects of the image data
- Provides greater flexibility and scalability for managing and storing large volumes of image data, as the storage location can be easily scaled up or down based on demand, and the images can be easily moved or replicated across different storage locations and systems.