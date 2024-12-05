# class Group_Activity_Temporal_Classifer(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers):
#         super(Group_Activity_Temporal_Classifer, self).__init__()
        
#         resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
#         self.feature_extraction = nn.Sequential(
#             *list(resnet50.children())[:-1], # remove fc layer
#             nn.Dropout(0.5)  
#         )
        
#         self.lstm = nn.LSTM(
#                             input_size=input_size,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True,
#                         )

#         self.fc =  nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(32, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, num_classes)
#         )
    
#     def forward(self, x):
#         # Input shape: (batch, 9, 3, 244, 244)
#         b, seq, c, h, w = x.shape
#         x = x.view(b * seq, c, h, w)  # (batch * 9, 3, 244, 244)

#         x = self.feature_extraction(x)  # (batch * 9, 2048, 1, 1)
#         x = x.view(b, seq, -1)  # (batch, 9, 2048)
        
#         x, (h, c) = self.lstm(x)  # x: (batch, 9 , hidden_size)
#         x = x[:, -1, :]  # (64, hidden_size)
#         x = self.fc(x)  # (64, num_classes)
        
#         return x