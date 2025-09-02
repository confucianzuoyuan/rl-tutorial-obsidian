from data_generator import DataGenerator

g = DataGenerator()

r = [g.decode(data) for data in g.get_batch_data(prefix=True)[1]][:10]
print(r)

r = [g.decode(data) for data in g.get_batch_data(prefix=False)[1]][:10]
print(r)
