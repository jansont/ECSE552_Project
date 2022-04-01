from dataloaders import get_iterators

historical_len = 7
batch_size = 16
pred_len = 1

train_dl, val_dl = get_iterators(
    historical_len = historical_len,
    pred_len = pred_len,
    batch_size = batch_size,
)


def __test__dl__():
    for batch in train_dl:
        (node_features, edge_index, edge_features, labels_x), labels_y = batch
        print(node_features.shape)
        print(edge_index.shape)
        print(edge_features.shape)
        print(labels_x.shape)
        print(labels_y.shape)
        # print(labels_x[0], labels_y[0])
        break
