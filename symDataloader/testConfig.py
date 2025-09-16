"""
Test configuration for TQA-Bench experiments.
"""

# Database scales to test
dbScales = ['128k']
# dbScales = ['8k']

# Table filter - only test these tables if array is not empty
# If empty, test all tables
tableFilter = []
# tableFilter = ['water_quality']

# Context truncate length
# If True, disables readTables tool, and truncates executePython output
limitContextGrowth = False
# limitContextGrowth = True

# Inject context junk
# If True, injects junk row into each table to fill up context.
injectContextJunk = False
# injectContextJunk = True # Remember to set limitContextGrowth to True if you use this

# save file suffix
saveFileSuffix = ''
# saveFileSuffix = '_limited' # Use if limitContextGrowth is True
# saveFileSuffix = '_limitedWithJunk' # Use if limitContextGrowth && injectContextJunk is True