with open('main_weap_ml.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix sklearn predict calls with verbose (RandomForest, etc tidak support verbose)
content = content.replace('self.model.predict(X, verbose=0)', 'self.model.predict(X)')
content = content.replace('self.model.predict(X_test, verbose=0)', 'self.model.predict(X_test)')
content = content.replace('self.model.predict(X_in, verbose=0)', 'self.model.predict(X_in)')

with open('main_weap_ml.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done! File patched successfully.')