
from rcl_interfaces.srv import GetParameters

class param_service_client:
	def __init__(self,controller_node,param_names,service_name):
		self.param_names = param_names
		self.service_name = service_name

		self.param_client = controller_node.create_client(GetParameters,self.service_name)
		self.param_future = None
		self.service_available = False

	def call_param_service_(self):
		if self.service_available:
			req = GetParameters.Request()
			req.names = self.param_names
			self.param_future = self.param_client.call_async(req)
		else:
			self.service_available = self.param_client.wait_for_service(timeout_sec=0.1)

	def get_params(self):

		if self.param_future is None:# service is not yet available
			self.call_param_service_()
		else:
			if self.param_future.done(): # service is available, and async request has received responses.
				return self.param_future.result()

		return [] # Either service is not available, or the request has not received responses.