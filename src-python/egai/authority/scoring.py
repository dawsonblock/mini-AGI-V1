class ScorerRegistry:
    """Runner-owned scorer registry. Benchmark candidates cannot submit scorer code."""
    def __init__(self):self._scorers={}
    def register(self,name,fn):
        if not name or name in self._scorers: raise ValueError('invalid/duplicate scorer')
        self._scorers[name]=fn
    def get(self,name):
        if name not in self._scorers: raise KeyError('unregistered scorer: '+name)
        return self._scorers[name]
    @classmethod
    def defaults(cls):
        r=cls();r.register('exact_match',lambda p,e:1.0 if str(p).strip()==str(e).strip() else 0.0);return r
