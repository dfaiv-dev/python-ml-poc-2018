
Create a new Docker Host VM

* Edit the `parameters.json` file
* Get the subscriptionId (name?) and resource group name to deploy to
* run the `deploy.sh` script
* Copy the `ubu-dkr-setup` script(s) to the vm.  Something like:
    * `scp -r ./ubu-dkr-setup/ [devusername]@$(dns_az_eastus pcs-dev-ml-dkrh-002a):~/`
* run `sudo bash setup.sh` on remote machine
