"""
Test configuration for TQA-Bench experiments.
"""

# Database scales to test
dbScales = ['8k']

# Table filter - only test these tables if array is not empty
# If empty, test all tables
tableFilter = []
# tableFilter = ['water_quality']

# Context truncate length
# If True, disables readTables tool, and truncates executePython output
limitContextGrowth = True

# Inject context junk
# If True, injects junk row into each table to fill up context.
injectContextJunk = False

# save file suffix
saveFileSuffix = ''
# saveFileSuffix = '_limited'
# saveFileSuffix = '_limitedWithJunk'