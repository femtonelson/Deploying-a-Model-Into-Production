def data_preprocess(input_df):
    # This function processes an input pandas dataframe and returns a numpy array with the reduced features
    # Columns expected : track_id, acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence

    # Filter out quantitative features and drop 'track_id'
    features = input_df.drop(columns=['track_id'])

    # Import the StandardScaler class and instanciate it
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Scale the quantitative features i.e replace each value of a feature x by [(value-mean(x))/std(x)]
    scaled_train_features = scaler.fit_transform(features)

    # Import PCA class
    from sklearn.decomposition import PCA

    # Perform PCA with the chosen number of components = 6 and project data onto components
    n_components = 6
    pca = PCA(n_components, random_state=10)
    pca.fit(scaled_train_features)
    pca_projection = pca.transform(scaled_train_features)
    # Return the reduced numpy array with n_components = 6 features
    return pca_projection