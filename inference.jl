using Flux, Plots, Zygote, BSON



X = [ [x/278, y/419] for x in 1:278,y in 1:419]


BSON.@load "juliapic16_3.bson" p16_p

layers = [ Dense(28,28,sin) for _ in 1:8]
cnet = Chain(Dense(2,28,sin), layers... , Dense(28,3))


Flux.loadparams!(cnet, p16_p) # loading the 16 bit weights back in the network to see the difference
prediction_16 = cnet.(X); # running the net on the scaled indices
predicted_image_16 = reshape(prediction_16, (:,419)); # reshaping the prediction
compressed_16 = [RGB.([pi[1], pi[2], pi[3]]...) for pi in predicted_image_16]

plot(compressed_16)


