from subprocess import Popen, PIPE

def write_arguments_to_file(args, filename):
	with open(filename, 'w') as f:
		f.wite('%s: %s\n' % (key, str(value)))

def store_revision_info(src_path, output_dir, arg_string):
	try:
		cmd = ['git', 'rev-parse', 'HEAD']
		gitproc = Popen(cmd, stdout = PIPE, cwd  src_path)
		(stdout, _) = gitproc.communicate()
		git_hash = stdout.strip()
